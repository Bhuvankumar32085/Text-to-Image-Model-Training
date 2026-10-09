"""Script to generate synthetic sample images for local testing and smoke tests without external downloads."""
import os
import struct
import zlib

def create_raw_png(filepath: str, width: int = 512, height: int = 512, color: tuple = (100, 150, 200)):
    """Generate a valid RGB PNG image without external dependencies."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Construct raw RGB image data with scanlines
    r, g, b = color
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # filter type 0 (None)
        # Create subtle gradient pattern
        for x in range(width):
            pr = min(255, max(0, int(r * (x / width) + (255 - r) * (y / height))))
            pg = min(255, max(0, int(g * (1.0 - x / width) + 50)))
            pb = min(255, max(0, int(b * (y / height) + 30)))
            raw_data.extend((pr, pg, pb))
    
    compressed_data = zlib.compress(bytes(raw_data), 6)
    
    # PNG Signature
    png_bytes = bytearray(b"\x89PNG\r\n\x1a\n")
    
    # IHDR Chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xffffffff
    png_bytes.extend(struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc))
    
    # IDAT Chunk
    idat_crc = zlib.crc32(b"IDAT" + compressed_data) & 0xffffffff
    png_bytes.extend(struct.pack(">I", len(compressed_data)) + b"IDAT" + compressed_data + struct.pack(">I", idat_crc))
    
    # IEND Chunk
    iend_crc = zlib.crc32(b"IEND") & 0xffffffff
    png_bytes.extend(struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc))
    
    with open(filepath, "wb") as f:
        f.write(png_bytes)

def main():
    base_dir = os.path.join("data", "raw", "sample_dataset", "images")
    os.makedirs(base_dir, exist_ok=True)
    
    colors = [
        (40, 120, 220),   # 1. mountain/lake
        (180, 80, 40),    # 2. cozy cabin
        (220, 30, 150),   # 3. cyberpunk
        (80, 190, 180),   # 4. villa pool
        (220, 180, 60),   # 5. golden retriever
        (60, 50, 110),    # 6. owl moonlight
        (200, 40, 40),    # 7. abstract oil
        (140, 140, 150),  # 8. minimalist ceramic
    ]
    
    for idx, col in enumerate(colors, start=1):
        filename = f"sample_{idx:02d}.png"
        filepath = os.path.join(base_dir, filename)
        create_raw_png(filepath, 512, 512, col)
        print(f"Generated {filepath} (512x512 PNG)")

if __name__ == "__main__":
    main()
