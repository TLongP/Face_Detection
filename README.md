# Face_Detection

# To Do List

- [x] create a model and train the model 
- [ ] enhance the model via add dropout, batchnormalization, ...
- [ ] add a function to create data for training
- [ ] enhance the model with face recognition

# Model Summary

![Alt text](pics/Model.png?raw=true "model")

This model use convolutional layer( no Dense) so it can work with all input_shapes which are greater than 120.

- Input is image shape (120,120,3)
- The ouput of VGG16 is (3,3,filters)
- The Conv_1 and Conv_2 have kernel size (3,3) and padding "valid" so it will returns a (1,1,filters)
- Conv_Output1 has kernel size (1,1) and padding "valid" with only 1 filters so it will returns the objectness of this "box"
- Conv_Output2 has kernel size (1,1) and padding "valid" with 4 filters, this represents the coordinates of the upper-left and botten-right corner of the box
Note that we use sigmoid function in both output layer!

## Output of the model

![Alt text](pics/Output.png?raw=true "output")

- If the Inputshape is (150,150,3) than the return values will be a tuple of shape (2,2,1),(2,2,4)
- So that the "window glide on the picture eaach 30 pixels" 
- But since this project is not a yolo model the model will return only "the first box"


# What next?

## 1. labeling for training

For this we have 3 possible ways.

- 1st: self labeling
- 2nd: use webcam and draw a rectangle on the image and the person persitions such that his/her face fits in the rectangle
- 3rd: taking the picture of your face behind a monoton background (or use "meet.google"), then take the part, which contains only the face (with augmentation) and add in 
a new picture and add the label depends on where the faces were added.

For the third choice we can do automatically.

## 2. Face Recognition Model

## 2.1 About the Model

Update our current model such that the model will recognice person in the database,although their are few pictures of them.

For this we only need the Conv_1 layer from our pretrained model.

Create pictures of persons wich we want to recognice in different folders.

Let **X** be the input image and **f(X)** be the output of Conv_1 (you can also add more layer after Conv_1 and freeze all the other layers)

Let **d** be the euclidean metric ie.
$$ d(f(X_1),f(X_2)):= \|f(X_1)-f(X_2)\|_2 $$

Let say that $X_1$ and $X_2$ are images of 2 different person and let $X_1'$ another image of $X_1$ then we want the function f such that

$$ d(f(X_1),f(X_1')) + C \leq d(f(X_2),f(X_1'))$$ 

where the constant $C>0$. This formula will say that "$X_1'$ is more the same to $X_1$ then to $X_2$. Ant the constant $C$ will help to differentiate the persons more.

## 2.2 Define the loss function
By the equation above we get

$$ d(f(X_1),f(X_1')) + C - d(f(X_2),f(X_1')) \leq 0$$ 

Then we define the loss function for the tuple $(X_1,X_2,X_1')$

$$ L:= max\{0,\ d(f(X_1),f(X_1')) + C - d(f(X_2),f(X_1'))\}$$

So minimize the function $L$ will give us the function $f$, which satisfies the inequality above. So for m data points we define the loss function

$$\mathbb{L}:= \frac{1}{m} \sum_{i=1}^m \big |\ \|f(X_{1,m})-f(X_{1,m}')\|_2 + C - \|f(X_{2,m})-f(X_{1,m}')\|_2\ \big|_+$$
where 
$$\big |x \big |_+= max\{0,x\}$$

## 2.3 Create training set
For example we have 3 persons, which we want to recognice. We save image of the first person in the folder 1 and so on. For example each folder will contains 10 images.

Then the tuple $(X_1,X_2,X_1')$ will be :

- $X_1$ and $X_1'$ will be the images of the same person, but they are different pictures.
- $X_2$ is picture of another person

So overall we have 4860 trainings pair.
Further more we can add a 4th folder which contains persons which are differnt to person 1,2 and 3. This folders must not contains picture of the same person but picture of this folder will only be taken as $X_2$!
