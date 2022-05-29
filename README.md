# Face_Detection

# To Do List

- [x] create a model and train the model 
- [ ] enhance the model via add dropout, batchnormalization, ...
- [ ] add custome callback for custome training
- [ ] add a function to create data for training

## Model Summary

![Alt text](assets/Model.png?raw=true "Title")

- The first Dense-Output layer returns objectness.
- The second returns coordinates for the corner( upper left and lower right coordinates)
