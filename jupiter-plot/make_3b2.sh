A="$1"
B="$2"
C="$3"
D="$4"
E="$5"
F="$6"
OUT="$7"
ffmpeg -i "$A" -i "$B" -i "$C" -i "$D" -i "$E" -i "$F" \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[v]
  " -map "[v]" -c:v libx264 -crf 18 -preset fast "$OUT"
