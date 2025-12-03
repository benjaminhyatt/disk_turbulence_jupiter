# PNG encoding with ffmpeg
# Options:
#  -y         Overwrite output
#  -f image2pipe    Input format
#  -vcodec png     Input codec
#  -r $8        Frame rate
#  -i $1        Input file 1 (top-left)
#  -i $2		Input file 2 (top-mid)
#  -i $3		Input file 3 (top-right)
#  -i $4		Input file 4 (bot-left)
#  -i $5        Input file 5 (bot-mid)
#  -i $6        Input file 6 (bot-right)
#  -f mp4       Output format
#  -vcodec libx254   Output codec
#  -pix_fmt yuv420p  Output pixel format
#  -preset slower   Prefer slower encoding / better results
#  -crf 20       Constant rate factor (lower for better quality)
#  -vf "scale..."   Round to even size
#  $7      		Output file
#!/bin/bash

function png2mp4(){
	ffmpeg \
    -y \
    -f image2pipe \
    -vcodec png \
    -framerate $8 \
    -i $1 \
	-i $2 \
	-i $3 \
	-i $4 \
	-i $5 \
	-i $6 \
	-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[v]" \
	-map "[v]" $5.mp4 \
    -f mp4 \
    -vcodec libx264 \
    -pix_fmt yuv420p \
	-preset slower \
    -crf 20 \
    -vf "scale=trunc(in_w/2)*2:trunc(in_h/2)*2" \
    $7
}

png2mp4 vorticity_nu_2em04_gam_0d0ep00_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_cadence.mp4 vorticity_nu_2em04_gam_8d5ep01_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_cadence.mp4 vorticity_nu_2em04_gam_2d4ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_cadence.mp4 vorticity_nu_2em04_gam_4d0ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_cadence.mp4 vorticity_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_cadence.mp4 vorticity_nu_2em04_gam_1d9ep03_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_cadence.mp4 vorticity_nu_2em04_gam_3b2_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0.mp4 10
