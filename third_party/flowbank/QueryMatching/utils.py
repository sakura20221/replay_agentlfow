"""Tiny stdlib helpers used by the QueryMatching model files."""
import json
import pickle


def loadjson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def savejson(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def loadpkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def savepkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
