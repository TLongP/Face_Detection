# Face_Detection

# To Do List

- [x] create a model and train the model 
- [ ] enhance the model via add dropout, batchnormalization, ...
- [ ] add a function to create data for training

## Model Summary

![Alt text](pics/Model2.png?raw=true "Title")

This model use convolutional layer( no Dense) so it can work with all input_shapes which are greater than 120.

- Input is image shape (120,120,3)
- The ouput of VGG16 is (3,3,filters)
- The Conv_1 and Conv_2 have kernel size (3,3) and padding "same" so it will returns a (3,3,filters)
- Conv_Output1 has kernel size (3,3) and padding "valid" with only 1 filters so it will returns the objectness of this "box"
- Conv_Output2 has kernel size (3,3) and padding "valid" with 4 filters, this represents the coordinates of the upper-left and botten-right corner of the box
Note that we use sigmoid function in both output layer!

