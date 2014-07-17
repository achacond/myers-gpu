###Threa-cooperative Myers implementation on GPU
==================================================

Example of MultiGPU timeline: [link](https://drive.google.com/file/d/0Bw2Nno1-eC2ISlNMSkF2bHI0WGs/edit?usp=sharing)

####Workload / Data:

#####Human Reference
Files:
|
[Reference]() |
[GEM FM-index](https://drive.google.com/file/d/0Bw2Nno1-eC2IX2lpRWVUUW9OdEU/edit?usp=sharing) |
[Myers Binary Reference](https://drive.google.com/file/d/0Bw2Nno1-eC2IZXlsLWxFb3FWMWs/edit?usp=sharing)
|

#####Candidates of 1Million queries (GEM output: .prof) 

Synthetic data (1M queries):
|
[100](https://drive.google.com/file/d/0Bw2Nno1-eC2IZk9heHozc0xIdjQ/edit?usp=sharing) |
[200](https://drive.google.com/file/d/0Bw2Nno1-eC2INnNIMzFRRm13ek0/edit?usp=sharing) |
[400](https://drive.google.com/file/d/0Bw2Nno1-eC2IeEpaQl9XUHF4TlU/edit?usp=sharing) |
[600](https://drive.google.com/file/d/0Bw2Nno1-eC2IeUtwZnNORGdqVDg/edit?usp=sharing) |
[800](https://drive.google.com/file/d/0Bw2Nno1-eC2Ib21SVFRVa0FoNVE/edit?usp=sharing) |
[1000](https://drive.google.com/file/d/0Bw2Nno1-eC2IU29ueWd2emxLdU0/edit?usp=sharing) |
|

Real data queries:
|
[Illumina](https://drive.google.com/file/d/0Bw2Nno1-eC2IZk9heHozc0xIdjQ/edit?usp=sharing) |
[Roche 454](https://drive.google.com/file/d/0Bw2Nno1-eC2IQnJJZkh6N2NtMWs/edit?usp=sharing) |
[Pacbio](https://drive.google.com/file/d/0Bw2Nno1-eC2IZzBNb1RBRDk2d28/edit?usp=sharing) |
[ION Torrent](https://drive.google.com/file/d/0Bw2Nno1-eC2IOHlTc3BHX29XeDg/edit?usp=sharing)
|


#####Example: Input (candidates) and Ouput (scores)
Files: 
|
[Candidates: 1000 Bases - 100K Queries - 11 Candidates per Query](https://drive.google.com/file/d/0Bw2Nno1-eC2IRWhyRVVnQlp4ZHc/edit?usp=sharing) |
[Results](https://drive.google.com/file/d/0Bw2Nno1-eC2ITWhmNFlNYlM4TVU/edit?usp=sharing)
|

#####Compile
    make gcandidates-GEM-1st
    make sample-0

#####Generate input files
    head -100000 1000.prof > 1000.100K.api.prof
    ./gcandidates-GEM-1st ../../data/1000.100K.api.prof

#####Run
Set the visible GPUs:

    export CUDA_VISIBLE_DEVICES=0,1,2

Run Myers in 3 GPUs and 6 buffers:

    ./TEST 0.2 ../../data/HG.4bits.ref ../../data/1000.100K.api.prof.100000.11.0.gem.qry ../../data/1000.100K.api.prof.100000.11.0.gem.qry ../../data/1000.100K.api.prof.100000.11.0.gem.qry ../../data/1000.100K.api.prof.100000.11.0.gem.qry ../../data/1000.100K.api.prof.100000.11.0.gem.qry ../../data/1000.100K.api.prof.100000.11.0.gem.qry
