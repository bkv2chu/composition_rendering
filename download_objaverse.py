import os
import argparse
import shutil
import objaverse

def main():
    parser = argparse.ArgumentParser(description="Download Objaverse objects and place them in a specific directory.")
    parser.add_argument("--download_dir", type=str, required=True, help="Target directory for the .glb files.")
    parser.add_argument("--num_objects", type=int, default=10, help="Number of objects to download.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel download processes.")
    args = parser.parse_args()

    # Create target directory
    target_dir = os.path.abspath(args.download_dir)
    os.makedirs(target_dir, exist_ok=True)

    print("Fetching UIDs...")
    all_uids = objaverse.load_uids()
    
    uids_to_download = all_uids[:args.num_objects] if args.num_objects > 0 else all_uids
    print(f"Downloading {len(uids_to_download)} objects to cache...")

    workers = args.num_workers if args.num_workers else os.cpu_count()

    # Step 1: Download to the default objaverse/huggingface cache
    objects = objaverse.load_objects(
        uids=uids_to_download,
        download_processes=workers
    )

    print(f"\nWriting files directly to {target_dir}...")
    
    # Step 2: Extract the actual .glb files and put them in your chosen directory
    success_count = 0
    for uid, cached_path in objects.items():
        if os.path.exists(cached_path):
            dest_path = os.path.join(target_dir, f"{uid}.glb")
            # Using copy2 preserves file metadata
            shutil.copy2(cached_path, dest_path)
            success_count += 1
            print(f"Saved: {dest_path}")
        else:
            print(f"Error: Could not find cached file for {uid} at {cached_path}")

    print(f"\nSuccessfully wrote {success_count} .glb files directly to: {target_dir}")

if __name__ == "__main__":
    main()
