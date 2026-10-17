"""MiniMax H3 clip chaining node.

[start image ->] clip 1 -> last frame -> clip 2 -> ... -> clip N -> merged video, inside one node.

Each clip is generated like the official MiniMax H3 image-to-video template (MiniMaxH3ImageToVideo ->
BasicGuider -> SamplerCustomAdvanced -> VAE decode video + audio) and written to
output/minimax_chain/<run_name>/ together with its last frame. Clips are cached on disk, so a
re-run only regenerates the clips whose inputs changed, and an interrupted run resumes.
"""

import hashlib
import json
import logging
import os
import re
from fractions import Fraction

import kornia
import numpy as np
import torch
from PIL import Image

import comfy.samplers
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import InputImpl, Types, io, ui
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

FPS = 24  # MiniMax H3 output frame rate
RUN_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def split_prompts(text):
    """Prompts are separated by a line containing only '---'."""
    blocks, current = [], []
    for line in text.splitlines():
        if line.strip() == "---":
            blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    blocks.append("\n".join(current))
    return [b.strip() for b in blocks if b.strip()]


def tensor_hash(image):
    return hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()


def save_png(image, path):
    Image.fromarray(np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)).save(path)


def load_png(path):
    # Same conversion as LoadImage. Chaining always goes through the saved PNG so the next
    # clip sees identical input whether the previous clip was just generated or cached.
    return torch.from_numpy(np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0)[None]


def match_chroma(image, reference, strength):
    """Match the Lab a/b (color) statistics of `image` to `reference`. Luminance is left untouched,
    so contrast and black point (freckles, shadows) keep the generated frame's values."""
    lab = kornia.color.rgb_to_lab(image.movedim(-1, 1).float())
    ref = kornia.color.rgb_to_lab(reference.movedim(-1, 1).float())[:, 1:]
    ab = lab[:, 1:]
    mean, std = ab.mean((2, 3), keepdim=True), ab.std((2, 3), keepdim=True).clamp_min(1e-6)
    matched = (ab - mean) * (ref.std((2, 3), keepdim=True) / std) + ref.mean((2, 3), keepdim=True)
    lab[:, 1:] = torch.lerp(ab, matched, strength)
    return kornia.color.lab_to_rgb(lab).clamp(0, 1).movedim(1, -1)


class MiniMaxH3ChainVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChainVideo",
            display_name="MiniMax H3 Chain Video",
            category="video/minimax",
            description="Generates N consecutive MiniMax H3 clips, each starting from the last frame of the previous one, "
                        "saves every clip and last frame to output/minimax_chain/<run_name>/ and returns the merged video. "
                        "Without a start image the first clip is text-to-video. "
                        "Finished clips are cached on disk: re-running only regenerates clips whose inputs changed.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),
                io.Image.Input("start_image", optional=True, tooltip="First frame of clip 1. Leave empty to generate clip 1 from text only."),
                io.String.Input("prompts", multiline=True,
                                tooltip="One prompt per clip, separated by a line containing only ---"),
                io.String.Input("style", multiline=True, default="", tooltip="Appended to every prompt."),
                io.Int.Input("num_clips", default=0, min=0, max=10000,
                             tooltip="0 = one clip per prompt. Fewer prompts than clips repeats the last prompt; extra prompts are ignored."),
                io.Float.Input("duration", default=5.0, min=0.2, max=150.0, step=0.1,
                               tooltip="Seconds per clip, snapped up to the model's 17k+5 frame grid (5 s = 124 frames)."),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Boolean.Input("vary_seed", default=True, tooltip="Use seed + clip index for each clip."),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="res_multistep"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="simple"),
                io.Int.Input("reanchor_every", default=0, min=0, max=10000,
                             tooltip="Anti-drift: re-inject the reference image every K clips (0 = off)."),
                io.Combo.Input("reanchor_mode", options=["target", "replace"], default="target",
                               tooltip="target: clips K, 2K... end on the reference (smooth). replace: clips K+1, 2K+1... start from it (hard cut)."),
                io.String.Input("run_name", default="chain", tooltip="Folder under output/minimax_chain/. Letters, digits, - and _."),
                io.Int.Input("redo_from", default=0, min=0, max=10000,
                             tooltip="Regenerate this clip and all following ones even if cached (0 = off). "
                                     "Needed after changing model, LoRA or VAE, which the disk cache does not track."),
                io.Float.Input("color_match", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Match the colors (hue and saturation, not brightness or contrast) of each chained start frame "
                                       "to the reference image before generating the next clip. Counters skin-tone and color drift across clips. 0 = off."),
                io.Image.Input("reference_image", optional=True,
                               tooltip="Anti-drift reference (default: start_image, or clip 1's last frame without a start image)."),
            ],
            outputs=[io.Video.Output(tooltip="All clips merged, without the duplicated frame at each seam.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, prompts, style, num_clips, duration, width, height,
                seed, vary_seed, steps, sampler_name, scheduler, reanchor_every, reanchor_mode, run_name, redo_from,
                color_match, start_image=None, reference_image=None) -> io.NodeOutput:
        if not RUN_NAME.match(run_name):
            raise ValueError(f"run_name '{run_name}' may only contain letters, digits, - and _")
        prompt_list = split_prompts(prompts)
        if not prompt_list:
            raise ValueError("No prompts: write one prompt per clip, separated by a line containing only ---")
        n = num_clips or len(prompt_list)
        if len(prompt_list) < n:
            logging.warning(f"MiniMax chain: {len(prompt_list)} prompts for {n} clips, clips {len(prompt_list) + 1}-{n} reuse the last prompt.")
            prompt_list += [prompt_list[-1]] * (n - len(prompt_list))
        elif len(prompt_list) > n:
            logging.warning(f"MiniMax chain: {len(prompt_list)} prompts for {n} clips, ignoring the last {len(prompt_list) - n}.")
        suffix = f"\n\n{style.strip()}" if style.strip() else ""

        subfolder = f"minimax_chain/{run_name}"
        run_dir = os.path.join(folder_paths.get_output_directory(), subfolder)
        os.makedirs(run_dir, exist_ok=True)
        manifest_path = os.path.join(run_dir, "manifest.json")
        manifest = {"clips": {}}
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

        length = max(5, round(duration * FPS))  # MiniMaxH3ImageToVideo snaps this up to 17k+5
        reference = reference_image if reference_image is not None else start_image
        if reference is not None:
            reference = reference[:1]
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
        pbar = comfy.utils.ProgressBar(n)

        prev_last = start_image[:1] if start_image is not None else None
        videos = []
        for i in range(1, n + 1):
            reanchor = reanchor_every and i > 1 and (i - 1) % reanchor_every == 0 and reanchor_mode == "replace"
            first = reference if reanchor else prev_last
            target = reference if reference is not None and reanchor_every and reanchor_mode == "target" and i % reanchor_every == 0 else None
            prompt = prompt_list[i - 1] + suffix
            clip_seed = seed + i - 1 if vary_seed else seed
            video_path = os.path.join(run_dir, f"clip_{i:03d}.mp4")
            last_path = os.path.join(run_dir, f"clip_{i:03d}_last.png")

            matched = color_match if i > 1 and not reanchor else 0.0
            key_parts = [prompt, clip_seed, steps, sampler_name, scheduler, width, height, length,
                         tensor_hash(first) if first is not None else None, tensor_hash(target) if target is not None else None]
            if matched:
                key_parts.append(["chroma", matched])
            key = hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()
            entry = manifest["clips"].get(str(i))
            cached = entry is not None and entry["key"] == key and os.path.exists(video_path) and os.path.exists(last_path)
            if cached and not (redo_from and i >= redo_from):
                logging.info(f"MiniMax chain: clip {i}/{n} cached")
            else:
                logging.info(f"MiniMax chain: generating clip {i}/{n} (seed {clip_seed})")
                if matched:
                    # The previous clip's last frame drifts in color; pull its hue and saturation back toward the reference.
                    first = match_chroma(first, reference, matched)
                cond, latent = MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length,
                                                             first_frame=first, last_frame=target).args
                guider = Guider_Basic(model)
                guider.set_conds(cond)
                samples = SamplerCustomAdvanced.execute(Noise_RandomNoise(clip_seed), guider, sampler, sigmas, latent).args[0]
                images = nodes.VAEDecode().decode(vae, samples)[0]
                audio = vae_decode_audio(audio_vae, samples)
                video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(FPS)))
                video.save_to(video_path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
                save_png(images[-1:], last_path)
                manifest["clips"][str(i)] = {"key": key, "seed": clip_seed, "prompt": prompt,
                                             "first_frame": ("start" if first is not None else "text") if i == 1 else ("anchor" if reanchor else "previous")}
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                del images, audio, samples, latent, cond, video

            clip_video = InputImpl.VideoFromFile(video_path)
            if i > 1 and not reanchor:
                # Drop the first frame (and 1/24 s of audio): it duplicates the previous clip's last frame.
                clip_video = clip_video.as_trimmed(1 / FPS, 0, strict_duration=False)
            videos.append(clip_video)
            prev_last = load_png(last_path)
            if reference is None:
                # Text-to-video start: anchor the rest of the chain to the first clip's last frame.
                reference = prev_last
            pbar.update(1)

        final_name = "final.mp4"
        InputImpl.VideoFromList(videos).save_to(os.path.join(run_dir, final_name), format=Types.VideoContainer.MP4)
        final = InputImpl.VideoFromFile(os.path.join(run_dir, final_name))
        return io.NodeOutput(final, ui=ui.PreviewVideo([ui.SavedResult(final_name, subfolder, io.FolderType.output)]))
