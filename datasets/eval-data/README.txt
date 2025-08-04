# Description of Data

global_voices.en - full 30,000 source-side sentences from GlobalVoices (in-domain data)
global_voices.ru - full 30,000 target-size sentences from GlobalVoices (in-domain data)
global_voices_eval.en - sub-sampled 3000 source-side sentences from GlobalVoices (in-domain data)
global_voices_eval.ru - sub-sampled 3000 target-side sentences from GLobalVoices (in-domain data)
reddit_eval.en - 3063 source side senteces from Reddit (shifted data)
reddit_eval.ru - 3063 target-side senteces from Reddit (shifted data)
reddit_eval_meta.tsv - anomaly annotation for Reddit data

joint_eval.domains  - contatenated and shuffled reddit_eval and global_voices_eval data domain labels (0 in-domain, 1 shifted)
joint_eval.en - source-side sentences for contatenated and shuffled reddit_eval and global_voices_eval data 
joint_eval.ru - target-side sentences for contatenated and shuffled reddit_eval and global_voices_eval data 


For more information, please read out paper https://arxiv.org/abs/2107.07455 and checkout of GitHub - https://github.com/yandex-research/shifts

For any questions, please raise and issue on GitHub.


