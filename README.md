myers-gpu
=========

myers-gpu



bin/myers4bitsGPU-col-titan 0.2 input/human_g1k_v37.cleaned.fasta.3101804739.4bits.ref input/1000.qry

cd src; make myers4bitsGPU-col neq=32 word=32 arch=titan sizeQuery=1000 profile=on funnel=on shuffle=on ballot=off line=ca th_block=128 th_sm=128; make install; cd ..;

diff input/regions.H.Sapiens.S.mason.454.1M.l800.recut.prof.1000000.11.800.bin.qry.res-4bits-rev2.gpu input/regions.H.Sapiens.S.mason.454.1M.l800.recut.prof.1000000.11.800.bin.qry.res-4bits-padding.cpu | head
