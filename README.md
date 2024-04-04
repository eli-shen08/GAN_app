# GAN_app
 My first GAN application trained on cifar_10

 # About Project
This is a small project that i have created where i have created and trained a GAN model .
The model was trained on 25,000 images from the cifar_10 dataset .
By no means are the generated images good but sometimes they do resemble something .
This model was trained on google colab t4 Gpu for 2 hours approximately .
It trained for 180 epochs , each epoch had 195 batch size .

Originally on CIFAR-10 the images are of size 32 X 32 X 3 but since they are very small i had to upscale them after prediction using interpolation and then serve them in the web application .

# HOW TO USE 
### Make sure you have a GPU in your system .
1. Clone the repository locally gor your machine .
2. Make sure you get such a structure .

![1](https://github.com/eli-shen08/GAN_app/assets/61158656/cf1e6f00-ef10-43e7-82d5-993449bf154f)

3. Install requirements.txt with pip . After that you are pretty much set up.
4. Now all you need to do is run the app.py file and click on the link .
   
   ![Screenshot (2)](https://github.com/eli-shen08/GAN_app/assets/61158656/e959f3de-b455-4885-aead-305cca880883)

5. If you have done everthing correctly a new tab should open up .

   ![Screenshot (3)](https://github.com/eli-shen08/GAN_app/assets/61158656/246ee354-a404-44ac-9c7c-332b95c31c96)


6. If you want to preddict a single image click on Preict Single .

   
![Screenshot (4)](https://github.com/eli-shen08/GAN_app/assets/61158656/3a511096-1b7a-41a2-a0f8-3c7791b93f51)

7. If you want to see the model generate multiple image you know what to do .

   
![Screenshot (5)](https://github.com/eli-shen08/GAN_app/assets/61158656/faa71e92-0c24-4c23-86aa-1f34a41dd128)


Enjoy.
Thank You
