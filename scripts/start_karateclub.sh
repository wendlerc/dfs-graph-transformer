bsub -G es_puesch -o /cluster/scratch/wendlerc/lsf_gtrans -n 5 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" 
python exp/evaluate/selfattn/karateclub.py --graph_file ../graphs/reddit_threads/reddit_edges.json --label_file graphs/reddit_threads/reddit_target.csv 
--name transformer

bsub -G es_puesch -o /cluster/scratch/wendlerc/lsf_gtrans -n 5 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" 
python exp/evaluate/selfattn/karateclub.py --graph_file ../graphs/reddit_threads/reddit_edges.json --label_file graphs/reddit_threads/reddit_target.csv 
--pretrain_flag --name transformer-pretrained

bsub -G es_puesch -o /cluster/scratch/wendlerc/lsf_gtrans -n 5 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" 
python exp/evaluate/selfattn/karateclub.py --graph_file ../graphs/twitch_egos/twitch_edges.json --label_file graphs/twitch_egos/twitch_target.csv
--name transformer

bsub -G es_puesch -o /cluster/scratch/wendlerc/lsf_gtrans -n 5 -W 48:00 -R "rusage[mem=30000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" 
python exp/evaluate/selfattn/karateclub.py --graph_file ../graphs/twitch_egos/twitch_edges.json --label_file graphs/twitch_egos/twitch_target.csv
--pretrain_flag --name transformer-pretrained

