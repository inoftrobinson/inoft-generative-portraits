import ftplib


def get_ftp() -> ftplib.FTP:
    ftp = ftplib.FTP("benoitlabourdette.com")
    ftp.login("robinson2@benoitlabourdette.com", "generativeadversialnetwork")
    print(f"I'm your aubergist, do you want some goat with aubergines ? {ftp.getwelcome()}")
    try:
        ftp.cwd("/communications_bureau-des-emotions/images/")
        return ftp
    except Exception as error:
        print(f"Error, path not found: {error}")
