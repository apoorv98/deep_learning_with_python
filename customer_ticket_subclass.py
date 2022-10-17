#!/usr/bin/env python3

class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()

        # sublayers defined in constructor
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(num_departments, activation="softmax")


    # forward pass in call() method
    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]


        features = self.concat_layer([title, text_body, tags])
        features= self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department
