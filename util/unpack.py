# RAG/TAG Tiger - unpack.py
# Copyright (c) 2024 Stuart Riffle
# github.com/stuartriffle/ragtag-tiger

import imghdr
import mimetypes
import os, email, hashlib, shutil, py7zr
from .files import cleanpath
from .extensions import archive_file_types, mime_file_types
from .lograg import lograg, lograg_verbose, lograg_error

shutil.register_unpack_format('7zip', ['.7z'], py7zr.unpack_7zarchive)

def unpack_mime(file_bytes, output_folder, container_file, container_type):
    """Unpack MIME container file parts into a folder"""

    if container_type == ".doc":
        try:
            data_as_text = file_bytes.decode("utf-8", errors="ignore")
            looks_like_mime = "MIME-Version" in data_as_text and "Content-Type" in data_as_text
            if not looks_like_mime:
                # Nice try, ancient MS Word binary file
                lograg_verbose(f"\tignoring non-MIME .doc file \"{cleanpath(container_file)}\"")
                return
        except:
            pass

    msg = email.message_from_bytes(file_bytes)
    part_counter = 0

    for part in msg.walk():
        filename_prefix = f"part-{part_counter:03d}"
        part_counter += 1
        part_type = part.get_content_type()
        part_maintype = part.get_content_maintype()
        part_content = None

        if part_maintype == "text":
            part_content = part.get_payload(decode=True)
            if isinstance(part_content, bytes):
                part_content = part_content.decode("utf-8")
            if part_type == "text/html":
                output_filename = filename_prefix + ".html"
            else:
                output_filename = filename_prefix + ".txt"
        else:
            part_content = part.get_payload(decode=True)
            filename = part.get_filename()

            if not filename:
                content_disposition = part.get('Content-Disposition', '')
                disposition_split = content_disposition.split(';')
                for disp in disposition_split:
                    if disp.strip().startswith('filename'):
                        filename = disp.split('=')[1].strip().strip('"')

            if not filename:
                guessed_extension = mimetypes.guess_extension(part_type)
                if guessed_extension:
                    filename = "guessed-mimetype" + guessed_extension

            if not filename:
                try:
                    # FIXME - this does not work (?)
                    image_type = imghdr.what(None, h=bytes(part_content))
                    if image_type:
                        filename = f"guessed-imghdr.{image_type}"
                except: pass

            output_filename = f"{filename_prefix}-{filename if filename else 'unknown'}"
            #part_content = part.get_payload(decode=True)
            #output_filename = f"{filename_prefix}-{part.get_filename()}"

        if part_content:
            file_path = os.path.join(output_folder, output_filename)
            with open(file_path, "wb") as f:
                if isinstance(part_content, str):
                    # If the content is a string (like HTML or plain text), encode it as UTF-8
                    f.write(part_content.encode("utf-8"))
                else:
                    # If the content is bytes (like an image or application/octet-stream), write it directly
                    f.write(part_content)

def unpack_container_to_temp(container_file, temp_folder):
    """Unpack a container file into a temporary folder"""

    true_name = cleanpath(container_file)
    name_hash = hashlib.md5(true_name.encode()).hexdigest()
    output_folder = os.path.join(temp_folder, os.path.basename(container_file) + f"-{name_hash}.temp")
    unpacked_files = []

    try:
        os.makedirs(output_folder, exist_ok=True)
        container_type = os.path.splitext(container_file)[1]

        if container_type in archive_file_types:
            shutil.unpack_archive(container_file, output_folder)

        elif container_type in mime_file_types:
            file_bytes = open(container_file, "rb").read()
            unpack_mime(file_bytes, output_folder, container_file, container_type)

        unpacked_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder)]

    except Exception as e: 
        lograg_error(f"failure unpacking \"{cleanpath(container_file)}\" into \"{os.path.normpath(output_folder)}\": {e}")
        try:
            lograg_verbose(f"\tremoving \"{os.path.normpath(output_folder)}\"...")
            shutil.rmtree(output_folder)
        except: pass

    return unpacked_files



