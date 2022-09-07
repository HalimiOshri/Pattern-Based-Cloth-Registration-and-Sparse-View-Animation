import lmdb
import pickle
import os
import time
import pdb

if not os.name == "nt":
    # fix limited fd problem on ubuntu
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if rlimit[0] < 4096:
        print("Adjust ulimit to 4096")
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def default_value_encoder(val):
    return pickle.dumps(val, protocol=4)


def default_value_decoder(*args, **kwargs):
    return pickle.loads(*args, **kwargs)


class LMDBDict:
    def __init__(
        self,
        path: str,
        name: str = "LMDBDict",
        value_enc_func=default_value_encoder,
        value_dec_func=default_value_decoder,
        max_tries=100,
        readonly=True,
    ):
        if readonly and not os.path.exists(path):
            raise FileNotFoundError(f"Missing LMDB path: {path}")
        elif not os.path.exists(path):  # write mode, going to write a new db
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.name = name
        self.path = path
        self.max_tries = max_tries
        self.readonly = readonly
        self.value_enc_func = value_enc_func
        self.value_dec_func = value_dec_func

        self.load_db()

    def load_db(self):
        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                self.env = None

                lmdb_args = dict(
                    path=self.path,
                    readonly=self.readonly,
                    max_readers=4096,
                    max_spare_txns=20,
                    lock=not self.readonly,
                )
                if not self.readonly:
                    lmdb_args["map_size"] = 2 ** 40

                self.env = lmdb.open(**lmdb_args)
                break
            except Exception as e:
                print(e)
                # Wait 1 second before trying to recover from failure.
                time.sleep(1)
        else:
            raise RuntimeError(f"Failed to load DB after {self.max_tries} tries.")

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k != "env"}
        d = copy.deepcopy(d)
        return d

    def __setstate__(self, d):
        self.__init__(**d)

    def get(self, key):
        return self.__getitem__(key)

    def reload_db(self):
        print(f"DB: {self.name} failed. Reloading...")
        self.load_db()

    def __getitem__(self, key):
        key = str(key).encode()

        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                with self.env.begin() as txn:
                    value = txn.get(key)
                break
            except Exception as e:
                print(e)
                self.reload_db()
        else:
            raise RuntimeError(f"Failed to get item after {self.max_tries} tries.")

        if value is None:
            raise KeyError(f"Missing key: {key} in {self.name}")
        return self.value_dec_func(value)

    def set_multiple(self, keys, values):
        assert len(keys) == len(values)
        keys = [str(key).encode() for key in keys]
        values = [self.value_enc_func(value) for value in values]

        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                with self.env.begin(write=True) as txn:
                    while len(keys) > 0:
                        txn.put(keys[0], values[0])
                        keys.pop(0)
                        values.pop(0)
                break
            except BaseException:
                self.reload_db()
        else:
            raise RuntimeError(f"Failed to set item after {self.max_tries} tries.")

    def __setitem__(self, key, value):
        key, value = str(key).encode(), self.value_enc_func(value)
        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                with self.env.begin(write=True) as txn:
                    txn.put(key, value)
                break
            except BaseException:
                self.reload_db()
        else:
            raise RuntimeError(f"Failed to set item after {self.max_tries} tries.")

    def __contains__(self, key):
        key = str(key).encode()
        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                with self.env.begin() as txn:
                    value = txn.get(key)
                return value is not None
            except BaseException:
                self.reload_db()
        else:
            raise RuntimeError(
                f"Failed to check membership after {self.max_tries} tries."
            )

    def __len__(self):
        return self.env.stat()["entries"]

    def __iter__(self):
        return self.keys()

    def keys(self):
        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                with self.env.begin() as txn:
                    with txn.cursor() as cursor:
                        for key in cursor.iternext(values=False):
                            yield key.decode()
                break
            except GeneratorExit:
                break
            except BaseException:
                self.reload_db()
        else:
            raise RuntimeError(
                f"Failed to enumerate keys after {self.max_tries} tries."
            )

    def values(self):
        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                with self.env.begin() as txn:
                    with txn.cursor() as cursor:
                        for value in cursor.iternext(keys=False):
                            yield self.value_dec_func(value)
                break
            except GeneratorExit:
                break
            except BaseException:
                self.reload_db()
        else:
            raise RuntimeError(
                f"Failed to enumerate values after {self.max_tries} tries."
            )

    def items(self):
        tries = 0
        while tries < self.max_tries:
            tries += 1
            try:
                with self.env.begin() as txn:
                    with txn.cursor() as cursor:
                        for key, value in cursor:
                            yield key.decode(), self.value_dec_func(value)
                break
            except GeneratorExit:
                break
            except BaseException:
                self.reload_db()
        else:
            raise RuntimeError(
                f"Failed to enumerate items after {self.max_tries} tries."
            )

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def __repr__(self):
        return f"{self.name}: {self.env.stat()}"


if __name__ == "__main__":
    db = "/mnt/captures/shihenw/CPM/BodyAnimation_model/20200207--1410--5067077--pilot--ASLbody--reducedbody/learn_tri/result_db/learn_tri/baseline_lodestar0_jointslocalencoder7rhmesh0_3stage_45pt/learn_tri_iter_098000.db"
    dic = LMDBDict(db)
    # A = dic['/mnt/projects/helltalk/charon/charon_v2_actual_recording_5_3_2019/controlroom_dani/vr/dani_alpp03_control_room_5_3_2019/cts-2019-05-03-09-42-12.vrs_037041']
    pdb.set_trace()
