"use client";

import React, { useState, useRef, ChangeEvent, DragEvent } from 'react';
import styles from './Dropzone.module.css';

interface DropzoneProps {
    onFilesSelected: (files: File[]) => void;
    disabled?: boolean;
}

const getAllFilesFromEntry = async (entry: any): Promise<File[]> => {
    if (entry.isFile) {
        return new Promise((resolve) => {
            entry.file((file: File) => resolve([file]));
        });
    } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const allEntries: any[] = [];
        const readBatch = (): Promise<void> => {
            return new Promise((resolve) => {
                reader.readEntries((entries: any[]) => {
                    if (entries.length > 0) {
                        allEntries.push(...entries);
                        readBatch().then(resolve);
                    } else {
                        resolve();
                    }
                });
            });
        };
        await readBatch();

        const childFilesResults = await Promise.all(
            allEntries.map((childEntry) => getAllFilesFromEntry(childEntry))
        );
        return childFilesResults.flat();
    }
    return [];
};

export default function Dropzone({ onFilesSelected, disabled }: DropzoneProps) {
    const [isDragActive, setIsDragActive] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) setIsDragActive(true);
    };

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);
    };

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = async (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);

        if (disabled) return;

        const items = Array.from(e.dataTransfer.items);
        if (items && items.length > 0) {
            const filePromises = items.map((item) => {
                if (item.kind === 'file') {
                    const entry = item.webkitGetAsEntry();
                    return entry ? getAllFilesFromEntry(entry) : Promise.resolve([]);
                }
                return Promise.resolve([]);
            });

            const fileArrays = await Promise.all(filePromises);
            const allFiles = fileArrays.flat();

            const validFiles = allFiles.filter(
                (file) => file.type.startsWith('image/') || file.type.startsWith('video/'),
            );

            if (validFiles.length > 0) {
                onFilesSelected(validFiles);
            }
        } else if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            // Fallback for browsers that don't support DataTransferItems
            const files = Array.from(e.dataTransfer.files);
            const validFiles = files.filter(
                (file) => file.type.startsWith('image/') || file.type.startsWith('video/'),
            );
            if (validFiles.length > 0) {
                onFilesSelected(validFiles);
            }
        }
    };

    const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            const files = Array.from(e.target.files);
            // Filter is handled by accept attribute but good to double check if needed
            onFilesSelected(files);
        }
    };

    const handleClick = () => {
        if (!disabled && inputRef.current) {
            inputRef.current.click();
        }
    };

    return (
        <div
            className={`${styles.dropzone} ${isDragActive ? styles.active : ''} ${disabled ? styles.disabled : ''}`}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={handleClick}
        >
            <input
                ref={inputRef}
                type="file"
                multiple
                accept="image/*,video/*"
                onChange={handleChange}
                className={styles.input}
                disabled={disabled}
            />
            <div className={styles.content}>
                <span className={styles.icon}>📁</span>
                <p className={styles.text}>
                    Drag & drop images/videos or folders here, or click to select
                </p>
                <p className={styles.subtext}>
                    Supports JPG, PNG, WEBP, MP4, WEBM
                </p>
            </div>
        </div>
    );
}
