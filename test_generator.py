class instance():
    def test_infer(self):
        for i in range(10):
            yield i

class generator():

    def __init__(self) -> None:
        self.gen_instance = instance()
    
    def predict_(self):
        self.gen = self.gen_instance.test_infer()

    def predict(self):
        res = next(self.gen)
        return res

g = generator()

g.predict_()

print(g.predict())
print(g.predict())
print(g.predict())
print(g.predict())
