# -*- coding: utf-8 -*-
"""
By Shahabedin Chatraee Azizabadi
An example of OOP for the office accessories class.
"""
import pandas as pd
import seaborn as sns
import os
import logging
# Creating a new logger
logger=logging.getLogger(__name__)


# Parent class
class Writing_ins(object):
    ## Class Attribute
    instrument='office accessories'
     # Initializer / Instance Attributes
    def __init__(self, name,price,madein):#the set of attributes are common between the instances of the current class
        self.name=name
        self.price=price
        self.madein=madein
        
        
        
#instance method
    def description(self):
        return "{} is made in {}".format(self.name,self.madein)   
    def consumes(self,material):
        return " {} uses {}".format(self.name,material)
 
# Child class
class pens(Writing_ins):
    def handwrite(self, Type):
        return "{} is written by {}".format(self.name,Type)
# Second Child class     
class Pencil(Writing_ins):
    def grading(self, tone):
        return "{} has a grading of {}".format(self.name,tone)
    instrument='Cheap office accessories'            
    
# Third Child class
class Papers(Writing_ins):
    def __init__(self,name,price,madein,size):
        # The use of super() function**********VERY IMPORTANT****************__________________
        super().__init__(name, price,madein)
        self.size=size
    def papequalit(self, quality_type):
        return "{} has a quality type of {}".format(self.name, quality_type)
  #*******************_________________________****_________________________*****__________      

        # Instantiate the Writing_ins  object    
Faber=Writing_ins('Faber-Castell',56,'Germany')
Uma=Writing_ins('Uma',29,'Germany')
Karas=pens('Karas kustume',75,'USA')
Parker=pens('Parker',100,'UK')
Lyreco=Pencil('Lyreco',2,'France')    
Lyreco_paper=Papers('Lyreco_paper',0.2,'France','A4')      
# Access the instance attributes
print("{} is {}euro and {} is {}euro.".format(
    Faber.name, Faber.price, Uma.name, Uma.price))
if Faber.instrument=="office accessories":
      print("{} is a {}!".format(Faber.name, Faber.instrument))
      
print(Faber.description())      
print(Uma.consumes("Fountain pen inks"))
print(Karas.handwrite("hand on the paper"))
print(Parker.consumes("Fountain pen inks"))
#The isinstance() function is used to determine if an instance is also an instance of a certain parent class!
print(isinstance(Karas, Writing_ins))
print(Lyreco.grading("HB"))
print (Lyreco_paper.papequalit("white and 80g"))
print(Lyreco_paper.size)
#Remember that child classes can also override attributes and behaviors from the parent class!
if Karas.instrument=="office accessories":
    print("{} is an {}!".format(Karas.name, Karas.instrument))
if Lyreco.instrument=="Cheap office accessories":    
        print("{} is a {}!".format(Lyreco.name, Lyreco.instrument))
else:
    print("It is not")

