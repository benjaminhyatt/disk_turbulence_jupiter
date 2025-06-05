# PNG encoding with ffmpeg
# Options:
#  -y         Overwrite output
#  -f image2pipe    Input format
#  -vcodec png     Input codec
#  -r $3        Frame rate
#  -i $1        Input files from cat command
#  -f mp4       Output format
#  -vcodec libx254   Output codec
#  -pix_fmt yuv420p  Output pixel format
#  -preset slower   Prefer slower encoding / better results
#  -crf 20       Constant rate factor (lower for better quality)
#  -vf "scale..."   Round to even size
#  $2         Output file
#!/bin/bash
shopt -s expand_aliases
source /home/bah2659/.bashrc
function png2mp4(){
  cat $1* | ffmpeg \
    -y \
    -f image2pipe \
    -vcodec png \
    -framerate $3 \
    -i - \
    -f mp4 \
    -vcodec libx264 \
    -pix_fmt yuv420p \
	-preset slower \
    -crf 20 \
    -vf "scale=trunc(in_w/2)*2:trunc(in_h/2)*2" \
    $2
}
#png2mp4 vorticity_nu_2em04_gam_3ep01_kf_2ep01_Nphi_512_Nr_256_ring_0/write_ vorticity_nu_2em04_gam_3ep01_kf_2ep01_Nphi_512_Nr_256_ring_0.mp4 10
#png2mp4 vorticity_nu_2em04_gam_3ep01_kf_5ep01_Nphi_1024_Nr_512_ring_0/write_ vorticity_nu_2em04_gam_3ep01_kf_5ep01_Nphi_1024_Nr_512_ring_0.mp4 10
png2mp4 vorticity_nu_5em05_gam_3ep01_kf_5ep01_Nphi_1024_Nr_512_ring_0/write_ vorticity_nu_5em05_gam_3ep01_kf_5ep01_Nphi_1024_Nr_512_ring_0.mp4 10
