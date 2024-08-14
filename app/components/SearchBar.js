// components/SearchBar.js

import React from 'react';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import { styled } from '@mui/material/styles';

const CustomTextField = styled(TextField)({
    '& .MuiOutlinedInput-root': {
        borderRadius: '4px', // Ajuste cette valeur pour obtenir l'arrondi désiré
    },
});

const SearchBar = ({ options, value, onChange, placeholder }) => {
    return (
        <Autocomplete
            multiple
            options={options}
            value={value}
            onChange={(event, newValue) => onChange(newValue)}
            renderTags={() => null} // Pour masquer les Chips
            renderInput={(params) => (
                <CustomTextField
                    {...params}
                    variant="outlined"
                    placeholder={placeholder}
                />
            )}
            className="mb-4 w-full"
            disableClearable // Pour désactiver la possibilité de tout effacer
        />
    );
};

export default SearchBar;
