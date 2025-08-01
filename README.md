# ai-nlp-stt-research

```
- Folder: src/pipeline
    + Perform forced alignment, and verify and clean the speech-to-text data. 
    Create speech-to-text data using voice cloning and text-to-speech.
        * s1_segment_audio_with_vad.py
        * s2_run_nemo_stt.sh, s2_run_stt_data.py
        * s1_segment_audio_with_vad.py
        * s2.1_prepare_data.ipynb
        * s3_run_align.sh
        * s3.1_extract_segments.ipynb
        * s4_run_nemo_stt.sh, s4_run_w2v2_stt.py
        * s5_run_re-verify_segments.ipynb
        * s6_stats_data.ipynb
        * s7_train.sh


- Folder: src/kaldi
    + Contain scripts to prepare data, training and testing forced alignment model with kaldi
    + Run:
        * ./s1_build_n_gram_lm.sh
        * ./s2_prepare_lang.sh
        * ./s3_make_mfcc_pitch_feature.sh
        * ./s4_run_mmi_gmm_mfcc_pitch.sh
        * ./s5_process_alignments.sh


- Folder: scripts 
    + Contain scripts to training and testing speech-to-text model, training n-gram model with kenlm
    + Check run.sh file
```