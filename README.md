# Mask-Detection


![image](https://user-images.githubusercontent.com/88517236/199560105-42c00750-bcfd-444a-8ac1-d27b81da87fd.png)


This project consists of 2 stages:
1st stage isFace Detection:
Input: Dataset including faces with and without masks

For each image in dataset do
Convert them to RGB
Detect the faces in the whole image and crop them 
resize the result faces into (128, 128) and feed them to the classification model 

end
Output: faces (128, 128)


2nd Stage Mask Detection:
Input: Dataset including one face with or without masks -- OR -- The Output Of Face Recognition
For each image in dataset do
Resizing images into (128, 128) (if necessary)
normalizing the image and converting them to array
Flattening them
end
Output: Either Masked face or Unmasked  
