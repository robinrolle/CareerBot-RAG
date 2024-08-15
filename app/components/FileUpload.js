'use client';
import React, { useState } from 'react';
import { FilePond, registerPlugin } from 'react-filepond';
import 'filepond/dist/filepond.min.css';
import '../style/FileUpload.css';
import FilePondPluginFileValidateType from 'filepond-plugin-file-validate-type';
import FilePondPluginFileValidateSize from 'filepond-plugin-file-validate-size';

registerPlugin(FilePondPluginFileValidateType, FilePondPluginFileValidateSize);

const FileUpload = ({ onFileUpload, onFileRemove }) => {
    const [files, setFiles] = useState([]);

    const handleInit = () => {
        console.log('FilePond has initialized');
    };

    const handleProcessFile = (error, fileItem) => {
        if (error) {
            console.error('Error processing file:', error);
        } else {
            const filename = JSON.parse(fileItem.serverId).filename;
            onFileUpload(filename);
        }
    };

    // TODO erase mock and call backend
    const handleUpload = (fieldName, file, metadata, load, error, progress, abort) => {
        const formData = new FormData();
        formData.append('file', file);
        // Mock server response
        setTimeout(() => {
            const mockResponse = {
                filename: `${Date.now()}-${file.name}`,
                message: 'File uploaded successfully',
            };
            console.log('Mock upload success:', mockResponse);
            load(JSON.stringify(mockResponse));
        }, 1000);

        return {
            abort: () => {
                console.log('Upload aborted');
                abort();
            }
        };
    };

    // TODO erase mock and call backend
    const handleRemoveFile = async (source, load, error) => {
        const filename = JSON.parse(source).filename;
        // Mock server response
        setTimeout(() => {
            console.log('Mock delete success for file:', filename);
            load();
            onFileRemove();
        }, 1000);
    };

    return (
        <div className="max-w-md mx-auto bg-white p-4 rounded-lg shadow-xl">
            <FilePond
                files={files}
                allowMultiple={false}
                acceptedFileTypes={['application/pdf']}
                credits={false}
                oninit={handleInit}
                onprocessfile={handleProcessFile}
                server={{
                    process: handleUpload,
                    revert: handleRemoveFile
                }}
                allowDrop={true}
                allowBrowse={true}
                allowPaste={true}
                labelIdle='Drop your resume PDF file or <span class="filepond--label-action">Browse</span>'
            />
        </div>
    );
};

export default FileUpload;
