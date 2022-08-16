from pygbx import Gbx, GbxType
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
import gym.spaces as spaces
import functools
import numpy as np

def mul(a,b) -> int:
    if type(a) == spaces.Dict:
        n = 0
        for i in a:
            n += prod(a[i].shape)
        a = n
    
    if type(b) == spaces.Dict:
        n = 0
        for i in b:
            n += prod(b[i].shape)
        print(b)
        b = n
    
    return a*b

def prod(iterable: tuple[int]) -> int:
    return functools.reduce(mul, iterable, 1) # need to be fix with a dict

speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
poss = spaces.Box(low=0.0, high=999999999, shape=(
    4,
    3,
))  # historic of position
track = {
    "name": spaces.Box(0, 99999999, (1, ), np.uint64),
    "rotation": spaces.Box(0, 6, (1, ), np.int32),
    "position": spaces.Box(-99999999, 99999999, (3, ), np.int64),
    "speed": spaces.Box(-99999999, 99999999, (1, ), np.int64),
    "flags": spaces.Box(0, 8000000000, (1, ), np.uint64),
    "skin": spaces.Box(0, 8000000000, (1, ), np.uint64),
}

obs_space = spaces.Tuple((speed, poss, spaces.Space((100, spaces.Dict(track)))))

print(sum(prod(s for s in space.shape) for space in obs_space)
    )

exit()

g = Gbx("C:\\Users\\Corentin\\Documents\\Trackmania2020\\Maps\\My Maps\\Sans nom.Map.Gbx")
print(g)
print(g.class_id)
# challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
b = g.find_raw_chunk_id(0x0304301F)
print(b)
b.pos -= 4
b.seen_loopback = True
g._read_node(GbxType.CHALLENGE, -1, b)

for cl in list(g.classes.values()) + list(g.root_classes.values()):
    print(cl)
    print(cl.__dict__)

challenge = g.get_class_by_id(GbxType.CHALLENGE)
if not challenge:
    quit()

print(len(challenge.blocks))

for block in challenge.blocks:
    print(block.__dict__)