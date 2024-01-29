
import os, email, hashlib
import shutil
from files import archive_file_types, mime_file_types


from logging import log, log_verbose, log_error
from files import cleanpath


def unpack_mime(file_bytes, output_folder, container_file, container_type):

    if container_type == ".doc":
        looks_like_binary = any(b < 32 or b > 127 for b in file_bytes)
        if looks_like_binary:
            # Unsupported old MS Word doc
            return

    msg = email.message_from_bytes(file_bytes, policy=email.policy.default)

    part_counter = 0
    for part in msg.walk():
        filename_prefix = f"part-{part_counter:03d}"
        part_counter += 1
        part_type = part.get_content_type()
        part_content = part.get_content()
        part_maintype = part.get_content_maintype()

        if part_maintype == "text":
            part_encoding = part.get_content_charset() or "utf-8"
            part_text = part_content.decode(part_encoding, errors="ignore")
            part_content = part_text.encode("utf-8", errors="ignore")

            if part_type == "text/html":
                output_filename = filename_prefix + ".html"
            else:
                output_filename = filename_prefix + ".txt"
        elif part_type == "application/octet-stream" or part_maintype == "image":
            output_filename = f"{filename_prefix}-{part.get_filename()}"
        else:
            log_verbose(f"\tignoring unrecognized MIME part of type \"{part_type}\" in \"{cleanpath(container_file)}\"")
            continue

        file_path = os.path.join(output_folder, output_filename)
        with open(file_path, "wb") as f:
            f.write(part_content)

def unpack_temp_container(container_file, temp_folder):
    """Unpack a container file into a temporary folder"""

    true_name = cleanpath(container_file)
    name_hash = hashlib.md5(true_name.encode()).hexdigest()
    output_folder = os.path.join(temp_folder, os.path.basename(container_file) + f"-{name_hash}.temp")
    unpacked_files = []

    try:
        os.makedirs(output_folder)
        container_type = os.path.splitext(container_file)[1]

        if container_type in archive_file_types:
            shutil.unpack_archive(container_file, output_folder)
        elif container_type in mime_file_types:
            file_bytes = open(container_file, "rb").read()
            unpack_mime(file_bytes, output_folder, container_type)

        unpacked_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder)]

    except Exception as e: 
        log_error(f"failure unpacking \"{cleanpath(container_file)}\" into \"{os.path.normpath(output_folder)}\": {e}")
        try:
            log_verbose(f"\tremoving \"{os.path.normpath(output_folder)}\"...")
            shutil.rmtree(output_folder)
        except: pass

    return unpacked_files
