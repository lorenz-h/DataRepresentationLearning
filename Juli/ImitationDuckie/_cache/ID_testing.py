from _utils.ID_utils import ThreadSaveCounter

n_points = ThreadSaveCounter(maxvalue=30)

for i in range(0, 28):
    n_points.increment()
    print(n_points.value)
print(n_points.reached_upper_limit())
for i in range(0, 5):
    n_points.increment()
    print(n_points.value)
print(n_points.reached_upper_limit())