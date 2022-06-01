Using this documentation to build the library : 
https://packaging.python.org/en/latest/tutorials/packaging-projects/   


BUILD : python3 (or py if on windows) -m build  
DISTRIBUTE :  python3 -m twine upload --repository testpypi dist/* --verbose  
username = __token__  
password = the pypi test api token, ask tristan faine.  

## Trying it on another pc :  
CREATE VENV : python3 -m venv venv  
ACTIVATE VENV : source venv/bin/activate or directly venv/Scripts/activate if on windows

# Using editable install mode (with venv)
pip install --editable $PATH  
pip uninstall $LIB-NAME

After modifying the library, it can be reloaded within an interpretor by doing importlib.reload(packagename), or imp.reload(packagename) if below python ver 3.4

# Using the repository via TestPyPi
INSTALL LIB : python3 -m pip install --index-url https://test.pypi.org/simple/  --extra-index-url https://pypi.org/simple/ liblisibility-TristanFaine  
UPGRADE LIB : python3 -m pip install --index-url https://test.pypi.org/simple/  --extra-index-url https://pypi.org/simple/ liblisibility-TristanFaine --upgrade  
USE LIB : "from liblisibility import X"

#Should be available here : https://test.pypi.org/project/liblisibility-TristanFaine/