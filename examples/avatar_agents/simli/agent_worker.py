import time
start_time = time.time()
request_process_st = time.time()
from typing import AsyncIterable
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
    VideoGenerator,
)
# Hard Imports
import numpy as np
import json
import requests
from collections import deque
import cv2
import os
import traceback
import asyncio
from typing import Optional, Union
# Fast API Imports
import signal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# Video Model related imports
import torch
from videogen import videogen

logger = logging.getLogger("trugen-avatar")

# Global Videogen
global_video_gen = videogen(
    "avatar-worker",
    height=1080,
    width=1920
)

class Huma1(VideoGenerator):
    def __init__(self, options: AvatarOptions, avatar_config):
        parallel_data_load_start_time = time.time()
        self._options = options
        self._audio_queue = asyncio.Queue[Union[rtc.AudioFrame, AudioSegmentEnd]]()
        self._audio_resampler: Optional[rtc.AudioResampler] = None
        # use AudioByteStream to chunk the audio frames to expected frame size
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            samples_per_channel=options.audio_sample_rate // options.video_fps,
        )
        self._frame_ts: deque[float] = deque()
        # Quality control
        self._quality_requested: str = "QUALITY_EXCELLENT"
        # Frame playout variables
        self._frame_counter: int = 0
        self._frame_direction: str = "forward"
        # Avatar Config
        self._avatar_config = avatar_config
        # Task and variable to load static frames in parallel
        self._static_frames: AsyncIterable[rtc.VideoFrame] = []
        # Load Video Generator
        global global_video_gen
        self._video_gen = global_video_gen
        # Pre-allocate the RGBA tensor
        self._vg_rgba_tensor = torch.zeros(
            4,
            int(self._options.video_height),
            int(self._options.video_width),
            dtype=torch.uint8,
            device="cuda",
        )
        self._vg_rgba_tensor[3] = 255
        # --- Parallelize direct_load_to_gpu and load_images_in_parallel ---
        import concurrent.futures

        def _load_static_frames_sync():
            asyncio.run(
                self._load_static_frames(
                    os.path.join("persistent-storage", self._avatar_config["avatar_id"], "static_frames")
                )
            )

        def _direct_load_to_gpu():
            try:
                self._video_gen.direct_load_synctalk_gpu(
                    os.path.join("persistent-storage", self._avatar_config["avatar_id"], "source"),
                    os.path.join("persistent-storage", self._avatar_config["avatar_id"], "model"),
                    no_of_infer=self._avatar_config["Total_inference_frames"],
                    start_frame=self._avatar_config["start_frame"],
                    is_face_enhancer_enabled=False
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error in direct_load_to_gpu: {e}")

        def _load_images_in_parallel():
            return self._video_gen.load_images_in_parallel(
                os.path.join("persistent-storage", self._avatar_config["avatar_id"], "static_frames"),
                max_workers=min(32, (os.cpu_count() or 1) * 2)
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_gpu = executor.submit(_direct_load_to_gpu)
            future_images = executor.submit(_load_images_in_parallel)
            future_static = executor.submit(_load_static_frames_sync)
            # future_audio_encoder = executor.submit(self._video_gen.load_audio_encoder)

            concurrent.futures.wait(
                [future_gpu, future_images, future_static],
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            self._vg_full_images, self._vg_full_images_tensors = future_images.result()
        # --- End parallelization ---
        print(f"Time taken to load all data: {time.time() - parallel_data_load_start_time}")

    # Method to load and populate static frames
    async def _load_static_frames(self, folder_path):
        """Load and process static frame images in parallel."""
        sf_start_time = time.time()
        # Get sorted image paths
        image_files = sorted(
            [f for f in os.scandir(folder_path) if f.name.endswith((".jpg", ".jpeg"))],
            key=lambda x: int(os.path.splitext(x.name)[0]),
        )

        if not image_files:
            raise ValueError("No images found in the specified folder.")

        sf_start_time = time.time()

        def create_frames_video(
            image_files,
            video_path: str,
            width: int,
            height: int,
            fps: int = 25,
        ):
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter")

            for f in image_files:
                img = cv2.imread(f.path)
                if img is None:
                    raise RuntimeError(f"Failed to read {f.path}")

                img = cv2.resize(
                    img,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )

                # mp4 expects BGR
                writer.write(img)

            writer.release()

        def load_static_frames_from_video(
            video_path: str,
            width: int,
            height: int,
        ):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Failed to open frames video")

            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Ensure correct size
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                # BGR → RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = np.ascontiguousarray(frame)

                frames.append(
                    rtc.VideoFrame(
                        width,
                        height,
                        rtc.VideoBufferType.RGBA,
                        frame.tobytes(),
                    )
                )

            cap.release()

            if not frames:
                raise RuntimeError("No frames decoded from video")

            return frames

        frames_dir = os.path.join(
            "persistent-storage",
            self._avatar_config["avatar_id"],
            "static",
            "frames",
        )

        video_path = os.path.join(
            "persistent-storage",
            self._avatar_config["avatar_id"],
            "static_frames",
            "frames.mp4",
        )

        # Create video if missing
        if not os.path.exists(video_path):
            print("Precompiled video not found — creating it")
            create_frames_video(
                image_files,
                video_path,
                self._options.video_width,
                self._options.video_height,
                fps=25,
            )

        # Load frames from video
        self._static_frames = load_static_frames_from_video(
            video_path,
            self._options.video_width,
            self._options.video_height,
        )

        print(
            f"Static frames loaded from video in "
            f"{time.time() - sf_start_time:.3f}s "
            f"({len(self._static_frames)} frames)"
        )

    def update_quality(self, quality: str = "QUALITY_EXCELLENT"):
        self._quality_requested = quality

    def _resize_video_frame(self, frame: rtc.VideoFrame) -> rtc.VideoFrame:
        # Based on requested quality set the target_width and target_height
        target_width = 640
        target_height = 360
        if self._quality_requested == "QUALITY_EXCELLENT":
            return frame
        elif self._quality_requested == "QUALITY_GOOD":
            target_width = 640
            target_height = 360
        elif self._quality_requested == "QUALITY_POOR":
            target_width = 320
            target_height = 180

        # Convert VideoFrame buffer → numpy
        frame_bytes = bytes(frame.data)
        img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (frame.height, frame.width, 4)  # RGBA
        )

        # Resize
        resized = cv2.resize(
            img, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

        # Create new VideoFrame
        return rtc.VideoFrame(
            target_width,
            target_height,
            rtc.VideoBufferType.RGBA,
            resized.tobytes(),
        )

    def _update_frame_state(self):
        # Update frame counter and direction
        if self._frame_direction == "forward":
            self._frame_counter += 1
            if self._frame_counter >= len(self._static_frames) - 1:
                self._frame_direction = "backward"
        else:
            self._frame_counter -= 1
            if self._frame_counter <= 0:
                self._frame_direction = "forward"

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Called by the runner to push audio frames to the generator."""
        await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        """Called by the runner to clear the audio buffer"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._audio_bstream.flush()

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """
        Generate a continuous stream of video and audio frames.

        Notes:
            - When the audio buffer is empty, idle (silent) video frames are produced.
            - Frame streaming speed is automatically paced to match the `video_fps` option.
            - When an `AudioSegmentEnd` is encountered, it is yielded to notify the runner
              that playback of the current audio segment is complete.
        """
        return self._video_generation_impl()

    def apply_fade(self,audio_bytes, num_channels, fade_ms, sample_rate):
        samples = np.frombuffer(audio_bytes, dtype=np.int16).copy()  # 👈 IMPORTANT

        fade_samples = int(sample_rate * fade_ms / 1000)
        fade_samples = min(fade_samples, len(samples) // num_channels)

        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        # Fade-in
        samples[:fade_samples] = (
            samples[:fade_samples].astype(np.float32) * ramp
        ).astype(np.int16)

        # Fade-out
        samples[-fade_samples:] = (
            samples[-fade_samples:].astype(np.float32) * ramp[::-1]
        ).astype(np.int16)

        return samples.tobytes()

    def make_silence_frame(self, sample_rate, num_channels, frame_ms=40):
        samples_per_channel = int(sample_rate * frame_ms / 1000)
        total_bytes = samples_per_channel * num_channels * 2  # PCM16
        return rtc.AudioFrame(
            data=b"\x00" * total_bytes,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )

    async def _video_generation_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        # Frame playout process
        WAS_SILENT = True
        VIDEO_FPS = self._options.video_fps
        VIDEO_DT = 1.0 / VIDEO_FPS
        TARGET_CHUNKS = 25
        AUDIO_FRAME_MS = 40
        next_tick = time.monotonic()
        pending_audio_frames: list[rtc.AudioFrame] = []
        pending_audio_bytes = b""
        # Silence Audio Frame
        sample_rate = self._options.audio_sample_rate
        num_channels = self._options.audio_channels
        bytes_per_sample = 2  # 16-bit PCM
        samples_per_channel = int(sample_rate * 40 / 1000)
        total_bytes = samples_per_channel * num_channels * bytes_per_sample
        silence_audio_frame = rtc.AudioFrame(
            data=b"\x00" * total_bytes,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )
        
        while True:
            try:
                # timeout has to be shorter than the frame interval to avoid starvation
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5 / self._options.video_fps
                )
                self._audio_queue.task_done()
            except asyncio.TimeoutError:
                if len(self._static_frames) > 0:
                    static_frame = self._static_frames[self._frame_counter]
                    yield static_frame
                    yield silence_audio_frame
                    self._frame_ts.append(time.time())
                    # Set silence frame state
                    WAS_SILENT = True
                    # Update frame state
                    self._update_frame_state()
                continue
            audio_frames: list[rtc.AudioFrame] = []
            if isinstance(frame, rtc.AudioFrame):
                # resample audio if needed
                if not self._audio_resampler and (
                    frame.sample_rate != self._options.audio_sample_rate
                    or frame.num_channels != self._options.audio_channels
                ):
                    self._audio_resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=self._options.audio_sample_rate,
                        num_channels=self._options.audio_channels,
                    )

                if self._audio_resampler:
                    for f in self._audio_resampler.push(frame):
                        audio_frames += self._audio_bstream.push(f.data)
                else:
                    audio_frames += self._audio_bstream.push(frame.data)
            else:
                if self._audio_resampler:
                    for f in self._audio_resampler.flush():
                        audio_frames += self._audio_bstream.push(f.data)

                audio_frames += self._audio_bstream.flush()
            
            for af in audio_frames:
                pending_audio_frames.append(af)
                pending_audio_bytes += bytes(af.data)
                if len(pending_audio_frames) == TARGET_CHUNKS:
                    pending_audio_bytes += self.make_silence_frame(sample_rate, num_channels).data
                    merged_audio = rtc.AudioFrame(
                        data=pending_audio_bytes,
                        sample_rate=af.sample_rate,
                        num_channels=af.num_channels,
                        samples_per_channel=len(pending_audio_bytes)
                        // (2 * af.num_channels),
                    )
                    print(f"Audio Frame: {merged_audio.duration}")
                    frame_push_time = time.time()
                    # Generate frames
                    _yield_frame_counter = 1
                    # Convert audio data to bytes
                    audio_frame_bytes = merged_audio.data.tobytes()
                    # Get audio features
                    aud_features = self._video_gen.create_loader(
                        audio_frame_bytes,
                        self._frame_counter,
                        self._frame_counter,
                        len(self._static_frames),
                    )
                    # Initial setup before inference
                    self._video_gen.aud_ind = -1
                    # process_until = aud_features.shape[0]
                    process_until = min(
                        aud_features.shape[0],
                        len(pending_audio_frames),
                    )
                    # Generating data and variables for generating video frames
                    aud_ind = -1
                    # Generate list of reference frames to use
                    gen_list, _frame_direction = self._video_gen.generate_body_list(
                        self._frame_counter,
                        process_until,
                        self._frame_direction,
                        len(self._static_frames) - 1,
                    )
                    for idx in range(0, process_until):
                        # Per Frame audio feature
                        aud_ind += 1
                        aud_feature = self.get_audio_features(
                            aud_features,
                            aud_ind
                        )
                        video_frame, chunked_audio_frame = self._generate_frame(gen_list[idx], aud_feature, pending_audio_frames[idx])
                        # yield video_frame
                        if video_frame:
                            yield video_frame
                        if chunked_audio_frame:
                            yield chunked_audio_frame
                        self._frame_ts.append(time.time())
                        print(f"Frame push time: {time.time() - frame_push_time}")
                        frame_push_time = time.time()
                    pending_audio_frames.clear()
                    pending_audio_bytes = b""
                else:
                    continue
            # send the AudioSegmentEnd back to notify the playback finished
            if isinstance(frame, AudioSegmentEnd):
                print("Audio Segment End")
                # Pad and generate frames for the pending audio frames to 1000ms
                target_duration = 1  # seconds
                sample_rate = af.sample_rate
                num_channels = af.num_channels
                bytes_per_sample = 2  # PCM16
                target_samples = int(sample_rate * target_duration)
                target_bytes = target_samples * num_channels * bytes_per_sample
                raw_audio = bytes(af.data)
                current_bytes = len(raw_audio)
                if current_bytes < target_bytes:
                    pad_bytes = target_bytes - current_bytes
                    last_sample = raw_audio[-2:]  # int16
                    raw_audio += last_sample * (pad_bytes // 2)
                padded_audio_frame = rtc.AudioFrame(
                    data=raw_audio,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=target_samples,
                )
                # Convert audio data to bytes
                audio_frame_bytes = padded_audio_frame.data.tobytes()
                # Get audio features
                aud_features = self._video_gen.create_loader(
                    audio_frame_bytes,
                    self._frame_counter,
                    self._frame_counter,
                    len(self._static_frames),
                )
                # Initial setup before inference
                self._video_gen.aud_ind = -1
                # process_until = aud_features.shape[0]
                process_until = min(
                    aud_features.shape[0],
                    len(pending_audio_frames),
                )
                # Add additional silent frames
                target_frames = int(target_duration * 1000 / AUDIO_FRAME_MS)  # 25
                missing = target_frames - len(pending_audio_frames)
                if missing > 0:
                    silence = self.make_silence_frame(sample_rate, num_channels)
                    pending_audio_frames.extend([silence] * missing)
                # Generating data and variables for generating video frames
                aud_ind = -1
                # Generate list of reference frames to use
                gen_list, _frame_direction = self._video_gen.generate_body_list(
                    self._frame_counter,
                    process_until,
                    self._frame_direction,
                    len(self._static_frames) - 1,
                )
                for idx in range(0, process_until):
                    frame_push_time = time.time()
                    # Per Frame audio feature
                    aud_ind += 1
                    aud_feature = self.get_audio_features(
                        aud_features,
                        aud_ind
                    )
                    video_frame, chunked_audio_frame = self._generate_frame(gen_list[idx], aud_feature, pending_audio_frames[idx])
                    # yield video_frame
                    if video_frame:
                        yield video_frame
                    if chunked_audio_frame:
                        yield chunked_audio_frame
                    self._frame_ts.append(time.time())
                    # print(f"Frame push time: {time.time() - frame_push_time}")
                pending_audio_frames.clear()
                pending_audio_bytes = b""
                yield AudioSegmentEnd()


    def get_audio_features(self, features, index):
        left = index - 8
        right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        #auds = torch.from_numpy(features[left:right])
        auds = features[left:right]
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds

    def _generate_frame(
        self, gen_list_item, aud_feature, chunked_audio_frame
    ):
        try:
            frame_prep_time = time.time()
            # Reshape and unsqueeze the audio feature to match the input shape of the render_from_batch function
            aud_feature=aud_feature.reshape(32,16,16)
            aud_feature=aud_feature.unsqueeze(0)
            output,mask,gfpgan_tensor=self._video_gen.inference_synctalk_2d(gen_list_item,aud_feature)
            output = output/255.0
            rend_tensor = output
            img_tensor=self._vg_full_images_tensors[gen_list_item]
            # Paste Landscape
            full_image = self._video_gen.paste_landscape(
                rend_tensor,
                img_tensor,
                self._avatar_config["x"],
                self._avatar_config["y"],
                self._avatar_config["target_width"],
                self._avatar_config["target_height"],
            )
            full_image = self._video_gen.tensor_to_image_cuda(full_image)
            # Update RGBA tensor directly on GPU
            updated_rgba = self._video_gen.update_rgba_tensor(
                self._vg_rgba_tensor, full_image
            )
            image_np_rgba = (
                updated_rgba.permute(1, 2, 0)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )
            buffer = image_np_rgba.tobytes()
            frame = rtc.VideoFrame(
                int(self._video_gen.width),
                int(self._video_gen.height),
                rtc.VideoBufferType.RGBA,
                buffer,
            )
            # print(f"Frame Prep and Push Time: {time.time() - frame_prep_time}")
            # Update frame state
            self._update_frame_state()
            return(
                frame,
                chunked_audio_frame
                if chunked_audio_frame
                else None,
            )
        except Exception:
            traceback.print_exc()
            return None, None

    async def _generate_frames(
        self, aud_features, audio_frame: rtc.AudioFrame | None, chunked_audio_frames
    ) -> AsyncIterable[rtc.VideoFrame]:
        gen_prep_time = time.time()
        
        # Reset Video Generator
        self._video_gen.reset()
        # # Convert audio data to bytes
        # audio_frame_bytes = audio_frame.data.tobytes()
        # # Get audio features
        # aud_features = self._video_gen.create_loader(
        #     audio_frame_bytes,
        #     self._frame_counter,
        #     self._frame_counter,
        #     len(self._static_frames),
        # )
        # Initial setup before inference
        self._video_gen.aud_ind = -1
        process_until = aud_features.shape[0]
        # Generate list of reference frames to use
        gen_list, _frame_direction = self._video_gen.generate_body_list(
            self._frame_counter,
            process_until,
            self._frame_direction,
            len(self._static_frames) - 1,
        )
        # aud_feature0 = self.get_audio_features(aud_features, 0)
        # aud_feature0 = aud_feature0.reshape(32, 16, 16).unsqueeze(0)
        # _ = self._video_gen.inference_synctalk_2d(
        #     gen_list[0],
        #     aud_feature0
        # )
        # # Ensure GPU work completes
        # torch.cuda.synchronize()
        # Generating data and variables for generating video frames
        aud_ind = -1
        # Process audio data
        # img_np = self._vg_full_images_tensors[0]
        print(f"Time taken to generate body list: {time.time() - gen_prep_time}")
        try:
            # Iterate and generate frames
            for idx in range(0, process_until):
                frame_prep_time = time.time()
                aud_ind += 1
                aud_feature = self.get_audio_features(
                    aud_features,
                    aud_ind
                )
                # Reshape and unsqueeze the audio feature to match the input shape of the render_from_batch function
                aud_feature=aud_feature.reshape(32,16,16)
                aud_feature=aud_feature.unsqueeze(0)
                output,mask,gfpgan_tensor=self._video_gen.inference_synctalk_2d(gen_list[idx],aud_feature)
                output = output/255.0
                rend_tensor = output
                img_tensor=self._vg_full_images_tensors[gen_list[idx]]
                # Paste Landscape
                full_image = self._video_gen.paste_landscape(
                    rend_tensor,
                    img_tensor,
                    self._avatar_config["x"],
                    self._avatar_config["y"],
                    self._avatar_config["target_width"],
                    self._avatar_config["target_height"],
                )
                full_image = self._video_gen.tensor_to_image_cuda(full_image)
                # Update RGBA tensor directly on GPU
                updated_rgba = self._video_gen.update_rgba_tensor(
                    self._vg_rgba_tensor, full_image
                )
                image_np_rgba = (
                    updated_rgba.permute(1, 2, 0)
                    .contiguous()
                    .detach()
                    .cpu()
                    .numpy()
                )
                buffer = image_np_rgba.tobytes()
                frame = rtc.VideoFrame(
                    int(self._video_gen.width),
                    int(self._video_gen.height),
                    rtc.VideoBufferType.RGBA,
                    buffer,
                )
                yield (
                    frame,
                    chunked_audio_frames[idx]
                    if idx < len(chunked_audio_frames)
                    else None,
                )
                # print(f"Frame Prep and Push Time: {time.time() - frame_prep_time}")
                # Update frame state
                self._update_frame_state()
        except Exception:
            traceback.print_exc()

    def _get_fps(self) -> float | None:
        if len(self._frame_ts) < 2:
            return None
        return (len(self._frame_ts) - 1) / (self._frame_ts[-1] - self._frame_ts[0])


# Method to push usage and conversation status in database
def post_usage(usage):
    try:
        # Get current environment
        url = "https://api.trugen.ai/v1/usage"
        headers = {
            "X-API-Key": "cd65626299e74e65917bd3e4ffd2d29c",
            "Content-Type": "application/json",
        }
        payload = json.dumps(usage)
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
    except:
        traceback.print_exc()

# Asyncio Event Shutdown Indicator
shutdown_event = asyncio.Event()
shutdown_event.set()
# # Asyncio Event to show wether the current instance is running or not
# is_running_event = asyncio.Event()

def get_avatar_config(id: str):
    path = f"persistent-storage/{id}/config.json"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Avatar config not found at: {path}")
    with open(path, "r") as f:
        return json.load(f)

# Main job request handler
@utils.log_exceptions(logger=logger)
async def handler(event):
    # is_running_event.set()  # Indicator for the current instance running
    # Start new job request
    input = event["input"]
    avatar_config = input.get("avatar", None)
    start_time = time.time()
    # Start new job request
    input = event["input"]
    # avatar_config = {
    #     "id": "Jennefier",
    #     "width": 1920,
    #     "height": 1080,
    #     "x": 582,
    #     "y": 36,
    #     "target_width": 701,
    #     "target_height": 701,
    # } # input.get("avatar", None)
    # Get avatar specific data from storage
    print(avatar_config)
    avatar_config = get_avatar_config(avatar_config.get("id"))
    print(avatar_config)
    lk_config = input.get("lk", None)
    # # connect to the room
    # room = rtc.Room()
    # await room.connect(lk_config["url"], lk_config["token"])
    # print(f"Time taken to connect to room: {time.time() - start_time}")
    async def connect_room(lk_config):
        room = rtc.Room()
        await room.connect(lk_config["url"], lk_config["token"])
        print(f"Time taken to connect to room: {time.time() - start_time}")
        return room

    def init_video_gen(avatar_options, avatar_config):
        return Huma1(avatar_options, avatar_config)

    # define the avatar options and start the runner
    avatar_options = AvatarOptions(
        video_width=avatar_config["width"],
        video_height=avatar_config["height"],
        video_fps=25,
        audio_sample_rate=16000,
        audio_channels=1,
    )
    # Connect to Livekit Room and Load Huma-1 in parallel
    loop = asyncio.get_running_loop()
    room_task = asyncio.create_task(connect_room(lk_config))
    video_gen_future = loop.run_in_executor(
        None,  # default ThreadPoolExecutor
        init_video_gen,
        avatar_options,
        avatar_config,
    )
    room, video_gen = await asyncio.gather(
        room_task,
        video_gen_future,
    )
    runner = AvatarRunner(
        room,
        audio_recv=DataStreamAudioReceiver(room),
        video_gen=video_gen,
        options=avatar_options,
    )
    # stop when disconnect from the room or the agent disconnects
    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            logging.info(f"Agent {participant.identity} disconnected, stopping worker")
            should_stop.set()

    @room.on("disconnected")
    def _on_disconnected():
        logging.info("Room disconnected, stopping worker")
        should_stop.set()

    try:
        should_stop = asyncio.Event()
        await runner.start()
        # run until stopped or the runner is complete/failed
        tasks = [
            asyncio.create_task(runner.wait_for_complete()),
            asyncio.create_task(should_stop.wait()),
        ]
        print(f"Request to Runner Start time: {time.time() - request_process_st}")
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Push session usage if Conversation ID exist
        conversation_id = input.get("conversation_id", None)
        if conversation_id and len(conversation_id) > 5:
            session_usage = {
                "conversation_id": input.get("conversation_id", None),
                "usage": {"total_duration": (time.time() - start_time)},
                "status": "ENDED",
            }
            post_usage(session_usage)
        await utils.aio.cancel_and_wait(*tasks)
        await runner.aclose()
        await room.disconnect()
        logger.info("avatar runner stopped")
        # Shutdown Indicator Flag
        shutdown_event.set()
        # Schedule shutdown after response is sent
        await shutdown()
        return {"status": "terminated", "refresh_worker": True}


# Define FastAPI Server
app = FastAPI()


@app.get("/health")
async def health_check():
    shutdown_event.set()
    await shutdown()
    return JSONResponse(status_code=200, content={"status": "healthy"})


# @app.get("/ready")
# async def ready_check():
#     if is_running_event.is_set():
#         raise HTTPException(status_code=503, detail="Not ready max concurrency reached")
#     return JSONResponse(status_code=200, content={"status": "ready"})

@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        None,
        global_video_gen.load_audio_encoder
    )

async def shutdown():
    """Wait a short time to let response finish, then shut down."""
    while not shutdown_event.is_set():
        asyncio.sleep(1)
    await asyncio.sleep(0.5)  # ensure response is sent before shutdown
    os.kill(os.getpid(), signal.SIGINT)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutdown initiated. Waiting for ongoing requests to complete...")
    # Wait for active requests to finish
    while not shutdown_event.is_set():
        await asyncio.sleep(0.5)
    logger.info("All requests completed. Proceeding with shutdown.")
    shutdown_event.set()


@app.get("/shutdown")
async def start_avatar(request: Request):
    shutdown_event.set()
    await shutdown()
    return {"status": "success", "message": "Avatar server stopped successfully"}


@app.post("/start")
async def start_avatar(request: Request):
    try:
        # Get request data
        data = await request.json()

        # Start handler to proceed
        # asyncio.create_task(handler(data))
        await handler(data)

        return {"status": "success", "message": "Avatar started successfully"}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"status": "error", "error": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
