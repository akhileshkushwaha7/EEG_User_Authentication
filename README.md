# EEG_User_Authentication
EEG-based user identification system using 1D-convolutional long short-term memory neural networks )
(https://www.sciencedirect.com/science/article/pii/S095741741930096X

This is the code implemented for EEG-based user authentication in Python. For 10 and 50 participants, I used my local machine, and for 109 participants, I ran this code using GPUs. Also I have tried replacing the relu with equivalent approximation function

**Overview of the proposed EEG-based biometric identification system:**
The proposed EEG-based biometric identification system, which consists of two phases: enrollment
phase and identification phase. In this system, all users’ EEG biometrics are learned and stored in a 1D-Convolutional LSTM neural
network, trained in the enrollment phase. The recorded EEG signals, either in enrollment phase or in identification phase, will be
pre-processed including batch normalization, and segmented into
1-second normalized signal recordings before being fed into the
1D-convolutional LSTM. The identify of the 1-second EEG signal
recording will be the output of the trained 1D-convolutional LSTM
in the identification phase. In the rest of the methodology section,
EEG signal pre-processing, network architecture, EEG dataset, training, and k-fold cross-validation will be explained in detail.

**Dataset Link:**
https://physionet.org/content/eegmmidb/1.0.0/



