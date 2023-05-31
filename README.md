# Yet Another Darcy Solver

Installation 

    pip install -r requirements.txt

Lancement des tests

    pytest -vvv tests

> Le répertoire `tests` doit etre un miroir de `yads`. En particulier, un fichie `test_*` est crée par 
> fichier de `yads`.

Avant de commiter 

    black .
    mypy .
    mypy yads
