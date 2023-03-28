class PlaneMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value):
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        x, y = key
        if not (0 <= x < self.len_x and 0 <= y < self.len_y):
            raise IndexError("Index out of range")
        return x, y


class ToroidalMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value):
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        x, y = key
        return x % self.len_x, y % self.len_y


class CylindricalMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value):
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        x, y = key
        if not 0 <= y < self.len_y:
            raise IndexError("Index out of range")
        return x % self.len_x, y


class SphericalMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)  # len_x should be equal to len_y
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value):
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        x, y = key
        if x == -1:
            return y, 0
        elif x == self.len_x:
            return y, self.len_y - 1
        elif y == -1:
            return 0, x
        elif y == self.len_y:
            return self.len_x - 1, x
        else:
            return x, y


class MobiusBandMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value):
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        x, y = key
        if not 0 <= y < self.len_y:
            raise IndexError("Index out of range")
        if x == -1:
            return self.len_x - 1, self.len_y - 1 - y
        elif x == self.len_x:
            return 0, self.len_y - 1 - y
        else:
            return x, y


class KleinBottleMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value):
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        x, y = key
        if x == -1:
            return self.len_x - 1, self.len_y - 1 - y
        elif x == self.len_x:
            return 0, self.len_y - 1 - y
        else:
            return x % self.len_x, y % self.len_y


class ProjectivePlaneMatrix:
    def __init__(self, data):
        self.data = data
        self.len_x = len(data)
        self.len_y = len(data[0])

    def __getitem__(self, key):
        x, y = self.get_true_index(key)
        return self.data[x][y]

    def __setitem__(self, key, value):
        x, y = self.get_true_index(key)
        self.data[x][y] = value

    def __iter__(self):
        for x in range(self.len_x):
            for y in range(self.len_y):
                yield self[(x, y)]

    def get_true_index(self, key):
        x, y = key
        if x == -1:
            return self.len_x - 1, self.len_y - 1 - y
        elif x == self.len_x:
            return 0, self.len_y - 1 - y
        elif y == -1:
            return self.len_x - 1 - x, self.len_y - 1
        elif y == self.len_y:
            return self.len_x - 1 - x, 0
        else:
            return x, y
