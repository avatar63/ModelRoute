functions = []
for i in range(3):
    def show(i=i):       # default argument evaluated NOW, this iteration, not later
        print(i)
    functions.append(show)

for f in functions:
    f()