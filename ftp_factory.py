import ftplib
from emotions_recognition.credentials import *


def get_ftp() -> ftplib.FTP:
    ftp = ftplib.FTP(FTP_URL)
    ftp.login(FTP_LOGIN, FTP_PASSWORD)
    print(f"I'm your aubergist, do you want some goat with aubergines ? {ftp.getwelcome()}")
    try:
        ftp.cwd("/communications_bureau-des-emotions/images/")
        return ftp
    except Exception as error:
        print(f"Error, path not found: {error}")
