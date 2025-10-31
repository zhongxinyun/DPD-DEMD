# DPD-DEMD
Official Implementation of "DPD-DEMD: Denoising Prior Guided Weakly Supervised Image-Domain Dual-Energy Material Decomposition"



### Method Overview

This method can be viewed as a **weakly supervised model ensemble framework** that integrates multiple denoising priors to guide the network learning process.
 It incorporates **global–local regularization losses** to constrain the network output, ensuring effective denoising **while preserving spatial resolution** and fine structural details.



### Training Notes

- The **weights** of the *global structural regularization loss* and *local smooth regularization loss* should be carefully adjusted according to your dataset.
- The **local smooth term** should **not be assigned too large a weight**, as excessive smoothness may lead to detail loss and global value shift.



### Empirical Observation

Experiments show that when trained on **high-resolution datasets**, this method tends to achieve **clearer structural preservation** and **better resolution fidelity** compared with typical unsupervised denoising methods such as **Noise2Sim**.



### Experiment Results

**SIEMENS Clinical Dataset Result (VNC + Iodine map)**

![fig4](C:\Users\admin\Desktop\fig4.jpg)

**SIEMENS Clinical Dataset Result ( 70kev VMI)**

![fig5](C:\Users\admin\Desktop\fig5.jpg)
