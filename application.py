from flask import Flask, request, render_template, session
import time
import random
import string

app = Flask(__name__)

# Set a secret key for sessions
app.secret_key = 'main'  # Replace with a secure key

# Default expiration time for the key in seconds (e.g., 5 minutes)
DEFAULT_KEY_EXPIRATION_TIME = 5 * 60  # 5 minutes

# Function to generate a random key
def generate_random_key(plaintext_length):
    random_key = ''.join(random.choices(string.ascii_letters + string.digits, k=plaintext_length))
    return random_key

# Function to encrypt plaintext using a random key
def encrypt(plaintext, expiration_time):
    key = generate_random_key(len(plaintext))
    ciphertext = ''.join(chr(ord(p) ^ ord(k)) for p, k in zip(plaintext, key))
    return ciphertext, key, expiration_time

# Function to decrypt ciphertext using the key
def decrypt(ciphertext, key):
    plaintext = ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(ciphertext, key))
    return plaintext

# Route to display home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle encryption
@app.route('/encrypt', methods=['POST'])
def encrypt_route():
    try:
        plaintext = request.form['plaintext']
        key_expiry = request.form.get('key_expiry', DEFAULT_KEY_EXPIRATION_TIME, type=int)
        
        if not plaintext.strip():
            raise ValueError("Plaintext cannot be empty.")

        # Encrypt the plaintext with the specified expiration time
        ciphertext, key, expiration_time = encrypt(plaintext, key_expiry)

        # Store key and timestamp in session for later decryption validation
        session['key'] = key
        session['key_timestamp'] = time.time()  # Timestamp when key was generated
        session['expiration_time'] = expiration_time

        return render_template(
            'index.html',
            plaintext=plaintext,
            ciphertext=ciphertext,
            key=key,
            action="Encryption",
            result="success",
            key_expiration_time=expiration_time  # Pass the expiration time to the front-end
        )
    except ValueError as e:
        return render_template('index.html', error=str(e), result="error")
    except Exception as e:
        return render_template('index.html', error=f"An unexpected error occurred: {e}", result="error")

# Route to handle decryption
@app.route('/decrypt', methods=['POST'])
def decrypt_route():
    try:
        ciphertext = request.form['ciphertext']
        key = request.form['key']
        if not ciphertext.strip() or not key.strip():
            raise ValueError("Both ciphertext and key must be provided.")

        # Check if the key is expired
        key_timestamp = session.get('key_timestamp', 0)
        current_time = time.time()
        expiration_time = session.get('expiration_time', DEFAULT_KEY_EXPIRATION_TIME)
        time_left = expiration_time - (current_time - key_timestamp)  # Calculate remaining time

        if time_left <= 0:
            raise ValueError("The key has expired. Please generate a new one.")

        # Check if the key matches the one stored in the session
        stored_key = session.get('key', '')
        if key != stored_key:
            raise ValueError("Invalid decryption key.")

        # Perform decryption
        plaintext = decrypt(ciphertext, key)

        return render_template(
            'index.html',
            ciphertext=ciphertext,
            key=key,
            plaintext=plaintext,
            action="Decryption",
            result="success",
            time_left=int(time_left),  # Pass remaining time to the front-end
        )
    except ValueError as e:
        return render_template('index.html', error=str(e), result="error")
    except Exception as e:
        return render_template('index.html', error=f"An unexpected error occurred: {e}", result="error")

if __name__ == "__main__":
    app.run(debug=False)
