A="$1"
B="$2"
OUT="$3"
ffmpeg -i "$A" -i "$B" \
  -filter_complex "[0:v][1:v]xstack=inputs=2:layout=0_0|w0_0[v]
  " -map "[v]" -c:v libx264 -crf 18 -preset fast "$OUT"
