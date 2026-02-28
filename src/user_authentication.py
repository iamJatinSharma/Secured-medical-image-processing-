"""
Module: user_authentication.py
Description: Simple user authentication using hashed passwords.
"""
import hashlib
import os


def hash_password(password):
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt + pwd_hash

def verify_password(stored_hash, password):
    salt = stored_hash[:16]
    pwd_hash = stored_hash[16:]
    check_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return pwd_hash == check_hash

if __name__ == "__main__":
    # Example usage:
    # hashed = hash_password('my_password')
    # print(verify_password(hashed, 'my_password'))
    pass
