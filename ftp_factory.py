import ftplib
from credentials import *


def get_ftp(extra_filepath_to_navigate_to=None) -> ftplib.FTP:
    ftp = ftplib.FTP(FTP_URL)
    ftp.login(FTP_LOGIN, FTP_PASSWORD)
    print(f"I'm your aubergist, do you want some goat with aubergines ? {ftp.getwelcome()}")
    try:
        ftp.cwd(f"/communications_bureau-des-emotions{'' if extra_filepath_to_navigate_to is None else extra_filepath_to_navigate_to}")
        return ftp
    except Exception as error:
        print(f"Error, path not found: {error}")
