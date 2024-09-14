'use client'; 

import React, { useState } from 'react';
import { FilePond, registerPlugin } from 'react-filepond';
import 'filepond/dist/filepond.min.css';
import '../style/FileUpload.css';
import FilePondPluginFileValidateType from 'filepond-plugin-file-validate-type';
import FilePondPluginFileValidateSize from 'filepond-plugin-file-validate-size';

// Registering FilePond plugins to validate file type and size.
registerPlugin(FilePondPluginFileValidateType, FilePondPluginFileValidateSize);

const FileUpload = ({ onFileUpload, onFileRemove }) => {
    const [files, setFiles] = useState([]); 
    // Manages the state for files being handled by FilePond.

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
    // Handles the processing of a file after it is uploaded. Extracts the filename from the server response and calls the onFileUpload callback.

    const handleUpload = (fieldName, file, metadata, load, error, progress, abort) => {
        const formData = new FormData();
        formData.append('file', file);

        const request = new XMLHttpRequest();
        request.open('POST', 'http://localhost:8000/upload');
        // Sends a file upload request to the specified server endpoint.

        request.upload.onprogress = (e) => {
            progress(e.lengthComputable, e.loaded, e.total);
        };

        request.onload = function () {
            if (request.status >= 200 && request.status < 300) {
                const response = JSON.parse(request.responseText);
                load(JSON.stringify({ filename: response.filename }));
            } else {
                error('Failed to upload file');
            }
        };
        // Handles the server response after uploading. On success, it provides the filename back to FilePond.

        request.onerror = function () {
            error('Error occurred during the upload');
        };

        request.send(formData);

        return {
            abort: () => {
                request.abort();
                abort();
            }
        };
    };
    // Supports abortion of the file upload if needed.

    const handleRemoveFile = async (source, load, error) => {
        const filename = JSON.parse(source).filename;
        try {
            const response = await fetch(`http://localhost:8000/delete/${filename}`, {
                method: 'DELETE'
            });
    
            if (!response.ok) {
                throw new Error('Failed to delete file');
            }
    
            setFiles((currentFiles) => currentFiles.filter(file => file.serverId !== source));
    
            load(); 
            onFileRemove();
        } catch (err) {
            error('Could not remove file');
        }
    };
    // Handles the removal of a file by sending a delete request to the server.

    return (
        <div className="max-w-md mx-auto bg-white p-4 rounded-lg shadow-xl">
            <FilePond
                files={files}
                onupdatefiles={setFiles}
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
    // Renders the FilePond component.
};

export default FileUpload;
