# CardioML_ParamEstimation:

## Noninvasive estimation of aortic hemodynamics and cardiac contractility using machine learning
This repository contains scripts for performing regression analysis to estimate major cardiovascular parameters using non-invasive clinical data. The parameters include aortic systolic blood pressure (aSBP), cardiac Output (CO), and end-systolic elastance (Ees).

**Abstract**

Cardiac and aortic characteristics are crucial for cardiovascular disease detection. However, non-invasive estimation of aortic hemodynamics and cardiac contractility is still challenging. This paper investigated the potential of estimating aortic systolic pressure (aSBP), cardiac output (CO ), and endsystolic elastance (Ees) from cuff-pressure and pulse wave velocity (PWV) using regression analysis.The importance of incorporating ejection fraction (EF) as additional input for estimating Ees was also assessed. The models, including Random Forest, Support Vector Regressor, Ridge, Gradient Boosting, were trained/validated using synthetic data (n = 4,018) from an in-silico model. When cuff-pressure and PWV were used as inputs, the normalized-RMSEs/correlations for aSBP, CO, and Ees (best-performing models) were 3.36 ± 0.74%/0.99, 7.60 ± 0.68%/0.96, and 16.96 ± 0.64%/0.37, respectively. Using EF as additional input for estimating Ees significantly improved the predictions (7.00 ± 0.78%/0.92). Results showed that the use of noninvasive pressure measurements allows estimating aSBP and CO with acceptable accuracy. In contrast, Ees cannot be predicted from pressure signals alone. Addition of the EF information greatly improves the estimated Ees. Accuracy of the model-derived aSBP compared to in-vivo aSBP (n = 783) was very satisfactory (5.26 ± 2.30%/0.97). Future in-vivo evaluation of COand Ees estimations remains to be conducted. This novel methodology has potential to improve the noninvasive monitoring of aortic hemodynamics and cardiac contractility.

<img width="1192" alt="Screenshot at Oct 19 18-00-15" src="https://github.com/Vicbi/CardioML_ParamEstimators/assets/10075123/b93c61cb-5286-4ae0-a107-295c82eb20dd">


**Original Publication**

For a comprehensive understanding of the methodology and background, refer to the original publication: Bikia, V., Papaioannou, T. G., Pagoulatou, S., Rovas, G., Oikonomou, E., Siasos, G., ... & Stergiopulos, N. (2020). Noninvasive estimation of aortic hemodynamics and cardiac contractility using machine learning. Scientific Reports, 10(1), 15015.

**Citation**

If you use this code in your research, please cite the original publication:

Bikia, V., Papaioannou, T. G., Pagoulatou, S., Rovas, G., Oikonomou, E., Siasos, G., ... & Stergiopulos, N. (2020). Noninvasive estimation of aortic hemodynamics and cardiac contractility using machine learning. Scientific Reports, 10(1), 15015. https://doi.org/10.1038/s41598-020-72147-8

**License**

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details.

This work was developed as part of a research project undertaken by the Laboratory of Hemodynamics and Cardiovascular Technology at EPFL (https://www.epfl.ch/labs/lhtc/).


Feel free to reach out at vickybikia@gmail.com if you have any questions or need further assistance!
