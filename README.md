# AdvPatch: Real-world attack on MTCNN face detection system

By Edgar Kaziakhmedov, Klim Kireev, Grigorii Melnikov, Mikhail Pautov and Aleksandr Petiushko

This is the code for AdvPatch research article. The video is available [here]().

## Abstract 

Recent studies proved that deep learning approaches achieve remarkable results on face detection task. On the other hand, the advances gave rise to a new problem associated with the security of the deep convolutional neural network models unveiling potential risks of DCNNs based applications. Even minor input changes in the digital domain can result in the network being fooled. It was shown then that some deep learning-based face detectors are prone to adversarial attacks not only in a digital domain but also in the real world. In the paper, we investigate the security of the well-known cascade CNN face detection system - MTCNN and introduce an easily reproducible and a robust way to attack it. We propose different face attributes printed on an ordinary white and black printer and attached either to the medical face mask or to the face directly. Our approach is capable of breaking the MTCNN detector in a real-world scenario.

## The repo

The repository is organized as follows:

* **input_img** stores all images to be used for training, should be colored with patch markers.
                A row in the grid must be same-colored. The color difference between the
                neighbouring marker rows must not be greater than 1;
* **mtcnn** provides with public [FaceNet implementation](https://github.com/davidsandberg/facenet) of 
            [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html);
* **utils** contains multi-patch manager;
* **weights** weights for MTCNN sub-networks taken from the public [FaceNet implementation](https://github.com/davidsandberg/facenet);
* **output_img** all generated patches will be stored here
                 (you also can try to convert it to B/W before printing).

The attack is implemented in **adversarial_gen.py** source file, in order to train the patches follow the guideline:
1. Set images (at least 5-6);
2. Specify patches parameters;
3. Specify losses.

The rest of the code is well-documented.

NOTE: paste yout own TensofFlow implementation of *resize_area_batch* function (INTER_AREA resize algorithm)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.