"use client";

import React, { useState, useRef, ChangeEvent, DragEvent } from 'react';
import styles from './Dropzone.module.css';

interface DropzoneProps {
    onFilesSelected: (files: File[]) => void;
    disabled?: boolean;
}

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

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragActive(false);

        if (disabled) return;

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
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
