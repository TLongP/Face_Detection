# Face_Detection

# To Do List

- [x] create a model and train the model 
- [ ] enhance the model via add dropout, batchnormalization, ...
- [ ] add a function to create data for training

## Model Summary

![Alt text](pics/Model.png?raw=true "model")

This model use convolutional layer( no Dense) so it can work with all input_shapes which are greater than 120.

- Input is image shape (120,120,3)
- The ouput of VGG16 is (3,3,filters)
- The Conv_1 and Conv_2 have kernel size (3,3) and padding "valid" so it will returns a (1,1,filters)
- Conv_Output1 has kernel size (1,1) and padding "valid" with only 1 filters so it will returns the objectness of this "box"
- Conv_Output2 has kernel size (1,1) and padding "valid" with 4 filters, this represents the coordinates of the upper-left and botten-right corner of the box
Note that we use sigmoid function in both output layer!

# Output of the model

![Alt text](pics/Output.png?raw=true "output")

- If the Inputshape is (150,150,3) than the return values will be a tuple of shape (2,2,1),(2,2,4)
- So that the "window glide on the picture eaach 30 pixels" 
- But since this project is not a yolo model the model will return only "the first box"
