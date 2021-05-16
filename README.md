# Dependency

This project depends on

- Tkinter (should be builtin)
- scipy
- numpy
- PIL
- numba

```
pip install -r requirements.txt
// or
pip install numpy
pip install scipy
pip install pillow
pip install numba
```

# Usage
```shell
python main.py
```

the experiemnt result could be found in playground.ipynb

# Others
Rendering could take a few second (4-6s)
Since I did not implement GPU part, so it cannot be real-time
interactive, but it still is better to use
 multigrid algorithm than naive solution