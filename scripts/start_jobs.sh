bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 24:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py  --no_features --name r2r-nofeat-entrywise2-c4s --overwrite '{"training" : {"mode" : "rnd2rnd_entry"}, "model" : {"encoder_class" : "DFSCodeEncoderEntryBERT", "cls_for_seq" : false}, "data" : {"n_iter_per_split" : 15}}'

bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 24:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py  --no_features --name r2r-nofeat-entrywise2 --overwrite '{"training" : {"mode" : "rnd2rnd_entry"}, "model" : {"encoder_class" : "DFSCodeEncoderEntryBERT"}, "data" : {"n_iter_per_split" : 15}}'

bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 24:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py --name r2r-entrywise2-c4s --overwrite '{"training" : {"mode" : "rnd2rnd_entry"}, "model" : {"encoder_class" : "DFSCodeEncoderEntryBERT", "cls_for_seq" : false}, "data" : {"n_iter_per_split" : 15}}'

bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 24:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py  --name r2r-c4s --overwrite '{"training" : {"mode" : "rnd2rnd"}, "model" : {"cls_for_seq" : false}}'

bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py  --no_features --name r2r30-nofeat-entrywise2-c4s --overwrite '{"training" : {"mode" : "rnd2rnd_entry"}, "model" : {"encoder_class" : "DFSCodeEncoderEntryBERT", "cls_for_seq" : false}}'

bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py  --no_features --name r2r30-nofeat-entrywise2 --overwrite '{"training" : {"mode" : "rnd2rnd_entry"}, "model" : {"encoder_class" : "DFSCodeEncoderEntryBERT"}}'

bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py  --no_features --name r2r30-nofeat-c4s --overwrite '{"training" : {"mode" : "rnd2rnd"}, "model" : {"cls_for_seq" : false}}'

bsub -G ls_krausea -o /cluster/scratch/wendlerc/lsf_gtrans -n 4 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" python exp/pretrain/selfattn/pubchem_plus_properties.py  --no_features --name r2r30-nofeat --overwrite '{"training" : {"mode" : "rnd2rnd"}}'



