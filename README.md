

Just a note for future users who are interested in build a webserver with Flask and Gunicorn, but meet error like:
In my case it is AttributeError: Can't get attribute 'Contrastive_learning_layer' on <module '__main__' from

some online available errors like: 

"AttributeError: Can't get attribute 'Net' on <module '__main__' despite adding class definition inline"

“AttributeError: Can't get attribute on <module '__main__' from 'manage.py'>”

“AttributeError: Can't get attribute 'tokenizer' on <module '__main__'>”

"Unpickling saved pytorch model throws AttributeError: Can't get attribute 'Net' on <module '__main__' despite adding class definition inline"

etc.

I would highly recommend this repository and I refer to this [link](https://stackoverflow.com/questions/74229936/getting-cant-get-attribute-getimages-on-module-main-from-usr-local-b) (REALLY APPRECIATED)

the solution is to add a few line beforet the app.run line 

```
# import the __main__ function
import __main__

# create the custom function  
class Contrastive_learning_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, enzy_embed, smiles_embed):
        xxxx
        return x
# add the function to __main__ modules as an attributes
__main__.Contrastive_learning_layer = Contrastive_learning_layer

if __name__ == '__main__':
    app.run()
```

generally, I find if you just define a function at the app.py file, it is fine and you do not need to add it as a attribute, but if it is a class function, you must add it as an attribute.

hope it can help you somehow. 
