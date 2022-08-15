from pygbx import Gbx, GbxType
import tmrl.config.config_constants as cfg

print(cfg.DATASET_PATH)
print(cfg.CHECKPOINT_PATH)
print(cfg.MODEL_PATH_WORKER)
print(cfg.PRAGMA_LIDAR)
print(cfg.PRAGMA_RNN)

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