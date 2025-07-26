# Python_Projects
This is a repository containing all my Python Projects

## Project Title: Cursor_Control

Description:
In this project, I developed a system that uses a webcam to control the mouse cursor through real-time hand tracking. By leveraging computer vision techniques, the application detects and tracks hand movements, allowing users to move the cursor and perform click actions using hand gestures. This hands-free interface demonstrates an alternative input method, offering potential applications in accessibility tools, touchless controls, and interactive installations.

## Project Title: Methanol_Production_Prediction_RF

Description:
This project uses an experimental Random Forest model to predict methanol production rates based on process parameters like gas composition and catalyst properties. Built on a two-month anonymized dataset, the model aims to identify optimal conditions for production. A catalyst code mapping file is included to assist with data interpretation. The model is still in the research phase and has not been deployed in a real-world setting.

## Project Title: Road_Detection_Binary_Segmentation_UNet

Description: 
This project implements a binary segmentation system using a custom U-Net model to detect roads in urban environments based on the Cityscapes dataset. The pipeline includes parallel preprocessing of side-by-side RGB and segmentation mask images, model training with BCEWithLogitsLoss and Dice Loss, and inference on both validation and user-supplied images. Key metrics such as IoU and Dice Coefficient are computed to evaluate performance. Mixed precision training is supported via the 🤗 Accelerate library, and configuration is modular through a dedicated CONFIG class. The model outputs binary masks, highlighting the road class, and supports visualization and inference for real-world deployment scenarios like autonomous driving and smart city mapping.