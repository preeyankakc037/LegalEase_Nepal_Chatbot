import sqlite3
import bcrypt

DB_FILE = "users.db"

def add_password_column_if_not_exists(cursor):
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    if "password" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN password TEXT")
        print("Added 'password' column to users table.")

def list_all_users():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email FROM users")
    users = cursor.fetchall()
    print("Users in DB:")
    for u in users:
        print(u)
    conn.close()

def update_user(user_id, new_username, new_email, new_password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    add_password_column_if_not_exists(cursor)

    # Check if email is already used by another user
    cursor.execute("""
        SELECT id FROM users WHERE email = ? AND id != ?
    """, (new_email, user_id))
    existing_user = cursor.fetchone()

    if existing_user:
        print(f"Error: The email '{new_email}' is already used by another user (ID: {existing_user[0]}).")
        conn.close()
        return False

    # Hash the password before saving
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    cursor.execute("""
        UPDATE users
        SET username = ?, email = ?, password = ?
        WHERE id = ?
    """, (new_username, new_email, hashed_password, user_id))

    conn.commit()
    conn.close()
    print(f"User ID {user_id} updated to: {new_username} | {new_email} | password updated")
    return True

def authenticate_user(username_or_email, input_password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Normalize input
    username_or_email = username_or_email.strip().lower()

    cursor.execute("""
        SELECT id, username, email, password FROM users
        WHERE LOWER(TRIM(username)) = ? OR LOWER(TRIM(email)) = ?
    """, (username_or_email, username_or_email))

    user = cursor.fetchone()
    conn.close()

    if user is None:
        print("Invalid details: user not found")
        return False

    user_id, username, email, hashed_password = user

    if hashed_password is None:
        print("No password set for this user")
        return False

    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')

    if bcrypt.checkpw(input_password.encode('utf-8'), hashed_password):
        print(f"Login successful! Welcome {username}")
        return True
    else:
        print("Invalid details: password does not match")
        return False

if __name__ == "__main__":
    # List all users first
    list_all_users()

    # Update user example (change these variables as needed)
    user_id_to_update = 1
    update_user(user_id_to_update, "Preeyanka", "preeyankakc.07@gmail.com", "p124")

    # Test login example
    authenticate_user("Preeyanka", "p124")
