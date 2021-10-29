import subprocess


if __name__ == '__main__':
    subprocess.run("python app.py & python bot.py", shell=True)
