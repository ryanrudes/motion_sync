## Installation
Install `ffmpeg` and `exiftool`.

## Video Preprocessing
This part can be run quickly with
```bash
chmod +x scripts/preprocess_videos.bash
./scripts/preprocess_videos.bash $VIDEO_DIR $GVHMR_PATH $GVHMR_OUTPUT_DIR
```

<details>
  <summary>Step-by-step Instructions</summary>
  
#### Step 1
Record video (the higher the framerate the better)

#### Step 2
Get the focal length of the camera used to capture the video.
```bash
F_MM=$(exiftool -G1 -a -s "$DEMO" \
  | awk '/\[Keys\][[:space:]]+CameraFocalLength35mmEquivalent/ {print $NF}' \
  | tail -n 1)
if [ -z "$F_MM" ]; then
  F_MM=$(exiftool -G1 -a -s "$DEMO" \
    | awk '/\[VideoKeys\][[:space:]]+FocalLengthIn35mmFormat/ {print $NF}' \
    | tail -n 1)
fi
echo "Focal length: ${F_MM} mm"
```

#### Step 3
Get the frame rate that the video was recorded in.
```bash
export FPS=$(exiftool -G1 -a -s "$DEMO.MOV" \
  | awk -F': ' '/VideoFrameRate/ {print $2}')
echo "Frame rate: $FPS fps"
```

#### Step 4
Convert the video to mp4 in full quality and (optional) cut the audio.
```bash
ffmpeg -i "$DEMO.MOV" -c:v libx264 -crf 0 -an "$DEMO.mp4" -r $FPS
```

#### Step 5
Run GVHMR on the video and download the results from `outputs/$

```bash
cd $GVHMR_DIR && python tools/demo/demo.py --video "$DEMO.mp4" -s --f_mm $F_MM --output_root $GVHMR_OUTPUT_DIR
```
</details>